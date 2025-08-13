package com.example.yoloapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Size
import android.view.View
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivityYolo : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var imageView: ImageView
    private lateinit var overlay: BoxOverlay
    private lateinit var fabPick: FloatingActionButton
    private lateinit var fabCamera: FloatingActionButton

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var yolo: YoloV8Tflite
    private lateinit var yuvToRgb: YuvToRgbConverter

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null

    private val reqCameraPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else finish()
    }

    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handlePickedImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        imageView = findViewById(R.id.imageView)
        overlay = findViewById(R.id.boxOverlay)
        fabPick = findViewById(R.id.fabPick)
        fabCamera = findViewById(R.id.fabCamera)

        yolo = YoloV8Tflite(assets)
        yuvToRgb = YuvToRgbConverter()
        cameraExecutor = Executors.newSingleThreadExecutor()

        fabPick.setOnClickListener { pickImage.launch("image/*") }
        fabCamera.setOnClickListener { switchToCamera() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            reqCameraPerm.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { ia ->
                    ia.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )
                // Mostrar cámara
                previewView.visibility = View.VISIBLE
                imageView.visibility = View.GONE
                overlay.setBoxes(emptyList())
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val bmp = yuvToRgb.toBitmap(imageProxy)
            val dets = yolo.run(bmp, confThr = 0.25f, keepOnlyHome = true)
            android.util.Log.d("YOLO", "Detections (cam): ${dets.size}")
            dets.take(3).forEachIndexed { i, d ->
                android.util.Log.d("YOLO", "#$i cls=${d.clsId} score=${"%.2f".format(d.score)} box=${d.box}")
            }
            val boxes = dets.map {
                val label = yolo.labels.getOrNull(it.clsId) ?: "cls ${it.clsId}"
                Box(it.box, label, it.score)
            }
            runOnUiThread { overlay.setBoxes(boxes) }
        } catch (t: Throwable) {
            t.printStackTrace()
        } finally {
            imageProxy.close()
        }
    }

    private fun switchToImageMode() {
        cameraProvider?.unbindAll()
        previewView.visibility = View.GONE
        imageView.visibility = View.VISIBLE
        overlay.setBoxes(emptyList())
    }

    private fun switchToCamera() {
        startCamera()
    }

    private fun handlePickedImage(uri: Uri) {
        switchToImageMode()
        val bmp = decodeUriToBitmap(uri, maxSize = 1600) ?: return
        imageView.setImageBitmap(bmp)

        val dets = yolo.run(bmp, confThr = 0.25f, keepOnlyHome = true)
        android.util.Log.d("YOLO", "Detections (img): ${dets.size}")

        val boxes = dets.map {
            val label = yolo.labels.getOrNull(it.clsId) ?: "cls ${it.clsId}"
            Box(it.box, label, it.score)
        }
        overlay.setBoxes(boxes)
    }

    private fun decodeUriToBitmap(uri: Uri, maxSize: Int = 1600): Bitmap? {
        return contentResolver.openInputStream(uri)?.use { stream ->
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            stream.mark(Int.MAX_VALUE)
            BitmapFactory.decodeStream(stream, null, bounds)
            stream.reset()

            val w = bounds.outWidth
            val h = bounds.outHeight
            var sample = 1
            val maxDim = maxOf(w, h)
            while ((maxDim / sample) > maxSize) sample *= 2

            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            BitmapFactory.decodeStream(stream, null, opts)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
