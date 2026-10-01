package com.example.handgesture

import android.content.Context
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt
import com.google.mediapipe.tasks.components.containers.Category

class GestureClassifierHelper(context: Context) {

    private val interpreter: Interpreter
    private val labels: List<String>

    init {
        interpreter = Interpreter(loadModelFile(context, "gesture_model.tflite"))
        labels = loadLabels(context, "labels.txt")
    }

    private fun loadModelFile(context: Context, path: String): MappedByteBuffer {
        val fd = context.assets.openFd(path)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    private fun loadLabels(context: Context, path: String): List<String> {
        return context.assets.open(path).bufferedReader().useLines { it.toList() }
    }

    private fun normalizeHand(landmarks: List<NormalizedLandmark>): FloatArray {
        val wristX = landmarks[0].x()
        val wristY = landmarks[0].y()
        val wristZ = landmarks[0].z()

        val translated = Array(21) { i ->
            floatArrayOf(
                landmarks[i].x() - wristX,
                landmarks[i].y() - wristY,
                landmarks[i].z() - wristZ
            )
        }

        val dx = translated[9][0]
        val dy = translated[9][1]
        var scale = sqrt(dx * dx + dy * dy)
        if (scale < 1e-6f) scale = 1f

        val result = FloatArray(63)
        for (i in 0 until 21) {
            result[i * 3] = translated[i][0] / scale
            result[i * 3 + 1] = translated[i][1] / scale
            result[i * 3 + 2] = translated[i][2] / scale
        }
        return result
    }

    private fun buildFeatureVector(result: HandLandmarkerResult): FloatArray {
        var leftHand: List<NormalizedLandmark>? = null
        var rightHand: List<NormalizedLandmark>? = null

        for (i in result.landmarks().indices) {
            val handednessForHand: List<Category>? = result.handedness().getOrNull(i)
            if (handednessForHand.isNullOrEmpty()) continue

            val side = handednessForHand[0].categoryName().lowercase()
            when (side) {
                "left" -> leftHand = result.landmarks()[i]
                "right" -> rightHand = result.landmarks()[i]
            }
        }
        val features = FloatArray(128)
        var idx = 0
        for (hand in listOf(leftHand, rightHand)) {
            if (hand != null) {
                features[idx++] = 1f
                val normalized = normalizeHand(hand)
                for (v in normalized) features[idx++] = v
            } else {
                idx += 64 
            }
        }
        return features
    }

    fun classify(result: HandLandmarkerResult): Pair<String, Float>? {
        if (result.landmarks().isEmpty()) return null

        val inputFeatures = arrayOf(buildFeatureVector(result))
        val output = arrayOf(FloatArray(labels.size))

        interpreter.run(inputFeatures, output)

        val probs = output[0]
        var maxIdx = 0
        for (i in probs.indices) {
            if (probs[i] > probs[maxIdx]) maxIdx = i
        }
        return labels[maxIdx] to probs[maxIdx]
    }

    fun close() {
        interpreter.close()
    }
}