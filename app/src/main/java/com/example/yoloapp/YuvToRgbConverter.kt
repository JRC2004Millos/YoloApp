package com.example.yoloapp

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

class YuvToRgbConverter {

    fun toBitmap(image: ImageProxy): Bitmap {
        val yPlane = image.planes[0].buffer
        val uPlane = image.planes[1].buffer
        val vPlane = image.planes[2].buffer

        val ySize = yPlane.remaining()
        val uSize = uPlane.remaining()
        val vSize = vPlane.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yPlane.get(nv21, 0, ySize)

        val chromaRowStride = image.planes[1].rowStride
        val chromaPixelStride = image.planes[1].pixelStride

        var offset = ySize
        for (row in 0 until image.height / 2) {
            var col = 0
            while (col < image.width / 2) {
                val uIndex = row * chromaRowStride + col * chromaPixelStride
                val vIndex = row * chromaRowStride + col * chromaPixelStride
                nv21[offset++] = vPlane.get(vIndex)
                nv21[offset++] = uPlane.get(uIndex)
                col++
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}
