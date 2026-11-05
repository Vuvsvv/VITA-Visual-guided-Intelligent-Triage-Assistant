package com.example.medicalaiguidance.service

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.text.TextPaint
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView

/**
 * Draws accessibility-node bounds without converting them to an app-window coordinate system.
 * AccessibilityNodeInfo#getBoundsInScreen() and this overlay both use physical screen coordinates.
 */
class OverlayManager(private val context: Context) {
    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private var overlayRoot: FrameLayout? = null
    private var borderView: View? = null
    private var bubbleView: TextView? = null
    private var isAdded = false

    fun show(rect: Rect, message: String? = null, messageAtTop: Boolean = false) {
        val screen = context.screenBounds()
        // Keep a small breathing room so the stroke does not cover the target's text,
        // while still making the frame read as tightly attached to the control.
        val padding = context.dp(6)
        val bubbleWidth = if (messageAtTop) {
            (screen.width() * 0.86f).toInt().coerceAtLeast(1)
        } else {
            screen.width()
        }
        val bubbleHeight = if (message.isNullOrBlank()) 0 else calculateBubbleHeight(message, bubbleWidth)
        val borderRect = Rect(rect).apply { inset(-padding, -padding) }.clampedTo(screen)
        val bubbleTop = if (messageAtTop) {
            context.dp(130).coerceIn(0, (screen.height() - bubbleHeight).coerceAtLeast(0))
        } else {
            (borderRect.top - bubbleHeight - context.dp(8)).coerceAtLeast(0)
        }
        val bubbleLeft = (screen.width() - bubbleWidth) / 2

        ensureViews()
        updateBorder(borderRect.left, borderRect.top, borderRect.width(), borderRect.height())
        updateBubble(message, bubbleWidth, bubbleHeight, bubbleLeft, bubbleTop)

        Log.d(
            "vgh_id_detect",
            "顯示紅框 node=${rect.toShortString()} border=${borderRect.toShortString()} " +
                "screen=${screen.width()}x${screen.height()} message=${message.orEmpty()}"
        )
        showRoot(screen)
        borderView?.post {
            val location = IntArray(2)
            borderView?.getLocationOnScreen(location)
            Log.d(
                "vgh_id_detect",
                "紅框實際左上=${location[0]},${location[1]} expected=${borderRect.left},${borderRect.top}"
            )
        }
    }

    fun showMessage(message: String) {
        val screen = context.screenBounds()
        val bubbleWidth = (screen.width() * 0.86f).toInt().coerceAtLeast(1)
        val bubbleHeight = calculateBubbleHeight(message, bubbleWidth)
        val bubbleLeft = (screen.width() - bubbleWidth) / 2
        val bubbleTop = context.dp(130).coerceIn(0, (screen.height() - bubbleHeight).coerceAtLeast(0))

        ensureViews()
        borderView?.visibility = View.GONE
        updateBubble(message, bubbleWidth, bubbleHeight, bubbleLeft, bubbleTop)
        Log.d("vgh_id_detect", "顯示提示泡泡 message=$message")
        showRoot(screen)
    }

    fun hide() {
        if (isAdded && overlayRoot != null) {
            runCatching { windowManager.removeView(overlayRoot) }
            isAdded = false
        }
    }

    private fun showRoot(screen: Rect) {
        val params = WindowManager.LayoutParams(
            screen.width(),
            screen.height(),
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            // Keep (0, 0) aligned with getBoundsInScreen(), including status/navigation-bar areas.
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 0
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                // R+ fits all system bars by default.  That moves an overlay below the
                // status bar even when FLAG_LAYOUT_IN_SCREEN is present, while node bounds
                // remain screen-relative.  Disable that automatic inset translation.
                setFitInsetsTypes(0)
                setFitInsetsSides(0)
                setFitInsetsIgnoringVisibility(false)
            }
        }

        runCatching {
            if (!isAdded) {
                windowManager.addView(overlayRoot, params)
                isAdded = true
            } else {
                windowManager.updateViewLayout(overlayRoot, params)
            }
            overlayRoot?.post {
                val location = IntArray(2)
                overlayRoot?.getLocationOnScreen(location)
                Log.d(
                    "vgh_id_detect",
                    "overlay 實際座標=${location[0]},${location[1]} " +
                        "size=${overlayRoot?.width}x${overlayRoot?.height} expected=0,0"
                )
            }
        }.onFailure { error ->
            Log.e("vgh_id_detect", "overlay 顯示失敗", error)
        }
    }

    private fun ensureViews() {
        if (overlayRoot != null) return

        overlayRoot = FrameLayout(context)
        borderView = View(context).apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = context.dp(10).toFloat()
                setStroke(context.dp(4), Color.RED)
                setColor(0x4DFF0000)
            }
        }
        bubbleView = TextView(context).apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = context.dp(9).toFloat()
                setColor(0xEE263238.toInt())
            }
            setTextColor(Color.WHITE)
            textSize = 16f
            gravity = Gravity.CENTER
            includeFontPadding = true
            minHeight = context.dp(36)
            setSingleLine(false)
            maxLines = 5
            setLineSpacing(context.dp(1).toFloat(), 1.0f)
            setPadding(context.dp(12), context.dp(6), context.dp(12), context.dp(6))
        }
        overlayRoot?.addView(borderView)
        overlayRoot?.addView(bubbleView)
    }

    private fun updateBorder(left: Int, top: Int, width: Int, height: Int) {
        borderView?.visibility = View.VISIBLE
        borderView?.layoutParams = FrameLayout.LayoutParams(width.coerceAtLeast(1), height.coerceAtLeast(1)).apply {
            leftMargin = left
            topMargin = top
        }
    }

    private fun updateBubble(message: String?, width: Int, height: Int, left: Int, top: Int) {
        if (message.isNullOrBlank()) {
            bubbleView?.visibility = View.GONE
            return
        }
        bubbleView?.visibility = View.VISIBLE
        bubbleView?.text = message
        bubbleView?.layoutParams = FrameLayout.LayoutParams(width.coerceAtLeast(1), height.coerceAtLeast(1)).apply {
            leftMargin = left
            topMargin = top
        }
    }

    private fun calculateBubbleHeight(message: String, bubbleWidth: Int): Int {
        val horizontalPadding = context.dp(24)
        val textWidth = (bubbleWidth - horizontalPadding).coerceAtLeast(1)
        val paint = TextPaint().apply {
            textSize = 16f * context.resources.displayMetrics.scaledDensity
        }
        val lines = message.split('\n').sumOf { line ->
            kotlin.math.ceil(paint.measureText(line.ifBlank { " " }) / textWidth).toInt().coerceAtLeast(1)
        }.coerceIn(1, 5)
        val lineHeight = (paint.fontMetrics.descent - paint.fontMetrics.ascent + context.dp(3)).toInt()
        return (context.dp(16) + lines * lineHeight).coerceAtLeast(context.dp(36))
    }
}

private fun Rect.clampedTo(screen: Rect): Rect =
    Rect(
        left.coerceIn(screen.left, (screen.right - 1).coerceAtLeast(screen.left)),
        top.coerceIn(screen.top, (screen.bottom - 1).coerceAtLeast(screen.top)),
        right.coerceIn((screen.left + 1).coerceAtMost(screen.right), screen.right),
        bottom.coerceIn((screen.top + 1).coerceAtMost(screen.bottom), screen.bottom)
    ).takeIf { it.width() > 0 && it.height() > 0 } ?: Rect(screen.left, screen.top, screen.left + 1, screen.top + 1)

private fun Context.screenBounds(): Rect =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        Rect((getSystemService(Context.WINDOW_SERVICE) as WindowManager).currentWindowMetrics.bounds)
    } else {
        @Suppress("DEPRECATION")
        DisplayMetrics().also { metrics ->
            (getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.getRealMetrics(metrics)
        }.let { Rect(0, 0, it.widthPixels, it.heightPixels) }
    }

private fun Context.dp(value: Int): Int =
    (value * resources.displayMetrics.density).toInt()
