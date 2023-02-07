package org.aa.seegrid

import nu.pattern.OpenCV
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs


fun main() {
    OpenCV.loadShared()
    loadImage("C:\\Users\\Antoine Alberti\\Pictures\\oim.jpg")
}

fun loadImage(imagePath: String): Mat = Imgcodecs.imread(imagePath)