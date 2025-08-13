package com.example.yoloapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

data class Box(
    val rect: RectF,
    val label: String,
    val score: Float
)

class BoxOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxes = mutableListOf<Box>()

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.GREEN
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
        style = Paint.Style.FILL
    }
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        alpha = 160
        style = Paint.Style.FILL
    }

    fun setBoxes(newBoxes: List<Box>) {
        synchronized(boxes) {
            boxes.clear()
            boxes.addAll(newBoxes)
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        synchronized(boxes) {
            for (b in boxes) {
                canvas.drawRect(b.rect, boxPaint)
                val text = "${b.label} ${(b.score * 100).toInt()}%"
                val tw = textPaint.measureText(text)
                val th = textPaint.fontMetrics.bottom - textPaint.fontMetrics.top
                val left = b.rect.left
                val top = (b.rect.top - th).coerceAtLeast(0f)
                canvas.drawRect(left, top, left + tw + 16f, top + th + 8f, bgPaint)
                canvas.drawText(text, left + 8f, top - textPaint.fontMetrics.top, textPaint)
            }
        }
    }
}
