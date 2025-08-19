package com.example.yoloapp

import android.content.res.AssetManager
import android.graphics.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

class YoloV8Tflite(
    private val assetManager: AssetManager,
    private val modelPath: String = "yolov8n_float32.tflite",
    private val labelsPath: String? = "labels.txt",
    private val inputSize: Int = 640
) {
    data class Detection(val clsId: Int, val score: Float, val box: RectF)

    val HOME_CLASS_IDS = setOf(56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75)

    private var gpuDelegate: GpuDelegate? = null

    private val interpreter: Interpreter by lazy {
        val bytes = assetManager.open(modelPath).readBytes()
        val buffer = ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder()).put(bytes).apply { rewind() }

        val options = Interpreter.Options()

        try {
            val compat = CompatibilityList()
            if (compat.isDelegateSupportedOnThisDevice) {
                // En TF Lite 2.13 usa el ctor sin opciones
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            } else {
                options.setNumThreads(4)
            }
        } catch (_: Throwable) {
            options.setNumThreads(4)
        }

        Interpreter(buffer, options)
    }

    fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
        try { gpuDelegate?.close() } catch (_: Throwable) {}
    }


    val labels: List<String> by lazy {
        try {
            labelsPath?.let {
                assetManager.open(it).use { s ->
                    s.readBytes().decodeToString().lines().filter { ln -> ln.isNotBlank() }
                }
            } ?: COCO
        } catch (_: Throwable) {
            COCO
        }
    }

    private data class Letterbox(val bmp: Bitmap, val scale: Float, val dx: Float, val dy: Float)

    private fun letterbox(src: Bitmap): Letterbox {
        val dstW = inputSize
        val dstH = inputSize
        val r = min(dstW.toFloat() / src.width, dstH.toFloat() / src.height)
        val newW = (src.width * r).toInt()
        val newH = (src.height * r).toInt()
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        val out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK)
        val dx = (dstW - newW) / 2f
        val dy = (dstH - newH) / 2f
        canvas.drawBitmap(resized, dx, dy, null)
        return Letterbox(out, r, dx, dy)
    }

    private fun bitmapToFloat32(bmp: Bitmap): ByteBuffer {
        val input = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(bmp.width * bmp.height)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        var i = 0
        for (y in 0 until bmp.height) {
            for (x in 0 until bmp.width) {
                val p = pixels[i++]
                input.putFloat(((p shr 16) and 0xFF) / 255f)
                input.putFloat(((p shr 8) and 0xFF) / 255f)
                input.putFloat((p and 0xFF) / 255f)
            }
        }
        input.rewind()
        return input
    }

    /** Asume export con NMS en grafo -> salida [1, N, 6] (cx,cy,w,h,score,cls) */
    fun run(
        src: Bitmap,
        confThr: Float = 0.35f,
        keepOnlyHome: Boolean = true
    ): List<Detection> {
        val lb = letterbox(src)
        val input = bitmapToFloat32(lb.bmp)

        val outShape = interpreter.getOutputTensor(0).shape() // ej. [1, 100, 6]
        val n = outShape[1]
        val out = Array(1) { Array(n) { FloatArray(6) } }

        interpreter.run(input, out)

        android.util.Log.d("YOLO", "INPUT shape=${interpreter.getInputTensor(0).shape().joinToString()} dtype=${interpreter.getInputTensor(0).dataType()}")
        android.util.Log.d("YOLO", "OUTPUT shape=${interpreter.getOutputTensor(0).shape().joinToString()} dtype=${interpreter.getOutputTensor(0).dataType()}")

        val dets = mutableListOf<Detection>()
        for (i in 0 until n) {
            val cx = out[0][i][0]
            val cy = out[0][i][1]
            val w  = out[0][i][2]
            val h  = out[0][i][3]
            val score = out[0][i][4]
            val clsId = out[0][i][5].toInt()

            if (score < confThr) continue
            if (keepOnlyHome && !HOME_CLASS_IDS.contains(clsId)) continue

            val x1 = cx - w/2f
            val y1 = cy - h/2f
            val x2 = cx + w/2f
            val y2 = cy + h/2f

            val rx1 = ((x1 - lb.dx) / lb.scale).coerceIn(0f, src.width.toFloat())
            val ry1 = ((y1 - lb.dy) / lb.scale).coerceIn(0f, src.height.toFloat())
            val rx2 = ((x2 - lb.dx) / lb.scale).coerceIn(0f, src.width.toFloat())
            val ry2 = ((y2 - lb.dy) / lb.scale).coerceIn(0f, src.height.toFloat())

            dets.add(Detection(clsId, score, RectF(rx1, ry1, rx2, ry2)))
        }
        return dets
    }

    // YoloV8Tflite.kt
    fun heatmapFromDetections(
        srcW: Int,
        srcH: Int,
        dets: List<Detection>,
        downsample: Int = 4,          // calcula en baja res y reescala
        sigmaFactor: Float = 0.25f,   // qué tan “anchas” son las manchas (relativo a la caja)
        alpha: Int = 0x66             // transparencia del overlay (0x00–0xFF)
    ): Bitmap {
        val w = maxOf(1, srcW / downsample)
        val h = maxOf(1, srcH / downsample)
        val field = FloatArray(w * h)

        for (d in dets) {
            val cx = ((d.box.left + d.box.right) * 0.5f) / downsample
            val cy = ((d.box.top + d.box.bottom) * 0.5f) / downsample
            val bw = (d.box.right - d.box.left) / downsample
            val bh = (d.box.bottom - d.box.top) / downsample

            val sigmaX = (bw * sigmaFactor).coerceAtLeast(1f)
            val sigmaY = (bh * sigmaFactor).coerceAtLeast(1f)
            val twoSigmaX2 = 2f * sigmaX * sigmaX
            val twoSigmaY2 = 2f * sigmaY * sigmaY

            val x0 = maxOf(0, kotlin.math.floor(cx - 3f * sigmaX).toInt())
            val x1 = minOf(w - 1, kotlin.math.ceil (cx + 3f * sigmaX).toInt())
            val y0 = maxOf(0, kotlin.math.floor(cy - 3f * sigmaY).toInt())
            val y1 = minOf(h - 1, kotlin.math.ceil (cy + 3f * sigmaY).toInt())

            var y = y0
            while (y <= y1) {
                val dy2 = (y - cy) * (y - cy)
                var x = x0
                while (x <= x1) {
                    val dx2 = (x - cx) * (x - cx)
                    val g = kotlin.math.exp(-dx2 / twoSigmaX2 - dy2 / twoSigmaY2).toFloat()
                    val idx = y * w + x
                    // usa max para que “gane” la mancha con mayor score
                    field[idx] = maxOf(field[idx], g * d.score)
                    x++
                }
                y++
            }
        }

        // normaliza a [0,1] y coloriza
        var mn = Float.POSITIVE_INFINITY; var mx = Float.NEGATIVE_INFINITY
        for (v in field) { if (v < mn) mn = v; if (v > mx) mx = v }
        val range = if (mx > mn) (mx - mn) else 1f

        val px = IntArray(w * h)
        for (i in field.indices) {
            val v = ((field[i] - mn) / range).coerceIn(0f, 1f)
            val c = jetColor(v) // colormap
            px[i] = ((alpha and 0xFF) shl 24) or c
        }
        val small = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        small.setPixels(px, 0, w, 0, 0, w, h)
        return Bitmap.createScaledBitmap(small, srcW, srcH, true)
    }

    private fun jetColor(v: Float): Int {
        val four = 4 * v
        val r = ((minOf(four - 1.5f, -four + 4.5f, 1f).coerceAtLeast(0f)) * 255).toInt()
        val g = ((minOf(four - 0.5f, -four + 3.5f, 1f).coerceAtLeast(0f)) * 255).toInt()
        val b = ((minOf(four + 0.5f, -four + 2.5f, 1f).coerceAtLeast(0f)) * 255).toInt()
        return ((r and 0xFF) shl 16) or ((g and 0xFF) shl 8) or (b and 0xFF)
    }

    companion object {
        val COCO = listOf(
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        )
    }
}
