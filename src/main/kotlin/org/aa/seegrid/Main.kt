package org.aa.seegrid

import javafx.application.Application
import javafx.scene.Scene
import javafx.scene.control.ScrollPane
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import javafx.scene.layout.FlowPane
import javafx.stage.Stage
import nu.pattern.OpenCV
import org.aa.seegrid.Color.*
import org.opencv.core.*
import org.opencv.core.CvType.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc.*
import org.opencv.ml.KNearest
import org.opencv.ml.Ml.ROW_SAMPLE
import org.opencv.objdetect.HOGDescriptor
import org.opencv.utils.Converters.*
import java.io.ByteArrayInputStream
import java.lang.Integer.max
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.extension
import kotlin.math.min


fun main() {
    OpenCV.loadShared()
    Application.launch(FxApp::class.java)
}

class FxApp : Application() {
    override fun start(primaryStage: Stage?) {
        val original = loadImage("src/main/resources/images/voisimage peluches.jpg")
            .resize(700)
        val threshold = original.toGrayScale().threshold()
        val contours = threshold.extractContours()
        val biggestContour = biggestContour(contours)
        val trapezoidContour = trapezoidContour(biggestContour)
        val deskewedImage = original.toRectangle(trapezoidContour)
        val cleanedUpDeskewed = deskewedImage.toGrayScale().threshold()
        val numberCandidates = cleanedUpDeskewed.extractContours().filterNumberCandidates()

        val digitClassifier = DigitClassifier()
        val horizontalLines = cleanedUpDeskewed.horizontalLines()
        val verticalLines = cleanedUpDeskewed.verticalLines()
        val positionedCandidates = positionContours(numberCandidates, verticalLines, horizontalLines)
        val regionsOfInterests = deskewedImage.regionsOfInterest(numberCandidates)
        val digits = digitClassifier.classify(regionsOfInterests.map { it.first })
        val boundedDigits = digits.zip(regionsOfInterests) { digit, imageAndRect -> Pair(digit, imageAndRect.second) }

        val numbersImage = cleanedUpDeskewed.black().drawContours(numberCandidates, PURPLE)
        val horizontalLinesImage = numbersImage.withLines(horizontalLines)
        val gridImage = horizontalLinesImage.withLines(verticalLines)
        val gridWithCoordinates = gridImage.withPositions(positionedCandidates)
        val redrawnGrid = deskewedImage.redrawGrid(boundedDigits, horizontalLines, verticalLines)
        initView(original, deskewedImage, gridWithCoordinates, redrawnGrid)
    }


    private fun Mat.redrawGrid(
        digits: List<Pair<String, Rect>>,
        horizontalLines: List<Pair<Point, Point>>,
        verticalLines: List<Pair<Point, Point>>
    ): Mat {
        val result = Mat(size(), CV_8UC3, Scalar(0.0, 0.0, 0.0))
        for (digit in digits)
            putText(
                result,
                digit.first,
                digit.second.bottomLeft(),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                GREEN.bgr
            )
        return result.withLines(horizontalLines).withLines(verticalLines)
    }

    private fun Mat.regionsOfInterest(numberCandidates: List<MatOfPoint>) =
        numberCandidates.toList()
            .map { boundingRect(it).plusMargin(2) }
            .map { Pair(submat(it), it) }

    private fun initView(vararg images: Mat) {
        val pane = FlowPane()
        pane.prefWrapLength = 1900.0
        pane.children.addAll(images.map { it.toFxImage().toImageView() })

        val stage = Stage()
        stage.scene = Scene(ScrollPane(pane))
        stage.width = 1900.0
        stage.height = 1000.0
        stage.show()
    }

    private fun Mat.toFxImage(): Image {
        val bytes = MatOfByte()
        Imgcodecs.imencode(".png", this, bytes)
        val inputStream = ByteArrayInputStream(bytes.toArray())
        return Image(inputStream)
    }

    private fun Image.toImageView(): ImageView {
        val imageView = ImageView()
        imageView.image = this
        return imageView
    }
}

private fun Mat.withPositions(positionedRectangles: List<PositionedRectangle>): Mat {
    val copy = copy()
    positionedRectangles
        .forEach {
            putText(
                copy,
                "(${it.row}, ${it.column})",
                Point(it.rectangle.x.toDouble(), (it.rectangle.y - 10).toDouble()),
                FONT_HERSHEY_SIMPLEX,
                0.4,
                RED.bgr
            )
        }
    return copy
}

private fun positionContours(
    contours: List<MatOfPoint>,
    verticalLines: List<Pair<Point, Point>>,
    horizontalLines: List<Pair<Point, Point>>
): List<PositionedRectangle> = contours
    .map { boundingRect(it) }
    .map { PositionedRectangle(column(it, verticalLines), row(it, horizontalLines), it) }

private fun Mat.copy(): Mat {
    val result = Mat()
    copyTo(result)
    return result
}

data class PositionedRectangle(val column: Int, val row: Int, val rectangle: Rect)

fun column(rectangle: Rect, verticalLines: List<Pair<Point, Point>>): Int {
    for (i in 0 until verticalLines.size - 1) {
        if (rectangle.center().x in verticalLines[i].first.x..verticalLines[i + 1].first.x)
            return i
    }
    return -1
}

fun row(rectangle: Rect, horizontalLines: List<Pair<Point, Point>>): Int {
    for (i in 0 until horizontalLines.size - 1) {
        if (rectangle.center().y in horizontalLines[i].first.y..horizontalLines[i + 1].first.y)
            return i
    }
    return -1
}

private fun loadImage(@Suppress("SameParameterValue") imagePath: String): Mat {
    return Imgcodecs.imread(imagePath)
}

private fun Mat.resize(maxWidth: Int): Mat {
    val ratio: Int = if (size().width > maxWidth) (size().width / maxWidth).toInt() else 1
    return resize(size().width / ratio, size().height / ratio)
}

private fun Mat.resize(width: Double, height: Double): Mat {
    val size = Size(width, height)
    val resized = Mat(size, type())
    resize(this, resized, size)
    return resized
}

private fun Mat.toGrayScale(): Mat {
    val destination = Mat(rows(), cols(), type())
    cvtColor(this, destination, COLOR_BGR2GRAY)
    blur(destination, destination, Size(2.0, 2.0))
    return destination
}

private fun Mat.threshold(): Mat {
    val destination = Mat()
    Core.bitwise_not(this, destination)
    adaptiveThreshold(destination, destination, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -10.0)
    return destination
}

private fun Mat.extractContours(): List<MatOfPoint> {
    val contours: List<MatOfPoint> = ArrayList()
    val hierarchy = Mat()
    findContours(this, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE)
    return contours
}

private fun biggestContour(contours: List<MatOfPoint>) = contours.maxBy { contourArea(it) }

private fun trapezoidContour(biggestContour: MatOfPoint) = approximate(biggestContour)

private fun approximate(contour: MatOfPoint): MatOfPoint {
    val contour2F = MatOfPoint2f()
    contour.convertTo(contour2F, CV_32FC2)
    val approximated2F = MatOfPoint2f()
    val perimeter = arcLength(contour2F, true)
    approxPolyDP(contour2F, approximated2F, 0.05 * perimeter, true)
    val approximatedContour = MatOfPoint()
    approximated2F.convertTo(approximatedContour, CV_32S)
    return approximatedContour
}

private fun Mat.toRectangle(trapezoid: MatOfPoint): Mat {
    val rectangle = MatOfPoint(
        Point(10.0, 10.0),
        Point(10.0, rows().toDouble() - 10.0),
        Point(cols().toDouble() - 10.0, rows().toDouble() - 10.0),
        Point(cols().toDouble() - 10.0, 10.0)
    )
    val transform = getPerspectiveTransform(trapezoid.toFloats(), rectangle.toFloats())
    val destination = copy()
    warpPerspective(this, destination, transform, destination.size())
    return destination
}

private fun Mat.horizontalLines(): List<Pair<Point, Point>> {
    val minWidth = (cols() / 25).toDouble()
    val horizontalContours = structure(Size(minWidth, 1.0)).extractContours()
    return horizontalContours
        .map { boundingRect(it) }
        .sortedBy { it.y }
        .fold(listOf()) { acc: List<Rect>, r: Rect ->
            when {
                acc.isEmpty() -> listOf(r)
                acc.last().isSameHorizontalLineAs(r) -> acc.dropLast(1) + union(acc.last(), r)
                else -> acc + r
            }
        }
        .map { Pair(Point(it.left().toDouble(), it.center().y), Point(it.right().toDouble(), it.center().y)) }
}

private fun Mat.verticalLines(): List<Pair<Point, Point>> {
    val minHeight = (rows() / 25).toDouble()
    val verticalContours = structure(Size(1.0, minHeight)).extractContours()
    return verticalContours
        .map { boundingRect(it) }
        .sortedBy { it.x }
        .fold(listOf()) { acc: List<Rect>, r: Rect ->
            when {
                acc.isEmpty() -> listOf(r)
                acc.last().isSameVerticalLineAs(r) -> acc.dropLast(1) + union(acc.last(), r)
                else -> acc + r
            }
        }
        .map { Pair(Point(it.center().x, it.top().toDouble()), Point(it.center().x, it.bottom().toDouble())) }
}

private fun Rect.isSameHorizontalLineAs(other: Rect) = center().isVerticallyBoundBy(other)
private fun Point.isVerticallyBoundBy(rectangle: Rect) = y >= rectangle.y - 5 && y <= rectangle.y + rectangle.height + 5

private fun Rect.isSameVerticalLineAs(other: Rect) = center().isHorizontallyBoundBy(other)
private fun Point.isHorizontallyBoundBy(rectangle: Rect) =
    x >= rectangle.x - 5 && x <= rectangle.x + rectangle.width + 5

private fun Mat.withLines(lines: List<Pair<Point, Point>>): Mat {
    val result = copy()
    for (line in lines) line(result, line.first, line.second, BLUE.bgr)
    return result
}

fun union(first: Rect, second: Rect): Rect =
    Rect(
        Point(min(first.left(), second.left()).toDouble(), min(first.top(), second.top()).toDouble()),
        Point(max(first.right(), second.right()).toDouble(), max(first.bottom(), second.bottom()).toDouble())
    )

private fun Rect.left() = x
private fun Rect.right() = x + width
private fun Rect.top() = y
private fun Rect.bottom() = y + height
fun Rect.center() = Point(x + width.toDouble().div(2), y + height.toDouble().div(2))
private fun Rect.bottomLeft() = Point(left().toDouble(), bottom().toDouble())
private fun Rect.plusMargin(margin: Int) = Rect(
    Point((left() - margin).toDouble(), (top() - margin).toDouble()),
    Point((right() + margin).toDouble(), (bottom() + margin).toDouble())
)

private fun Mat.structure(structureSize: Size): Mat {
    val destination = clone()
    val structure = getStructuringElement(MORPH_RECT, structureSize)
    erode(destination, destination, structure, Point(-1.0, -1.0))
    dilate(destination, destination, structure, Point(-1.0, -1.0))
    return destination
}

private fun MatOfPoint.toFloats(): MatOfPoint2f {
    val destination = MatOfPoint2f()
    convertTo(destination, CV_32F)
    return destination
}

private fun MatOfPoint2f.toInts(): MatOfPoint {
    val destination = MatOfPoint()
    convertTo(destination, CV_32S)
    return destination
}

private fun List<MatOfPoint>.filterNumberCandidates(): List<MatOfPoint> = map { it.toFloats() }
    .filter {
        val boundingRect = boundingRect(it)
        boundingRect.width in 4..20 && boundingRect.height in 12..20
    }.map { it.toInts() }

private fun Mat.drawContours(contours: List<MatOfPoint>, color: Color): Mat {
    val result = copy()
    drawContours(result, contours, -1, color.bgr)
    return result
}

private fun Mat.black(): Mat = Mat.zeros(size(), CV_8UC3)

enum class Color(val bgr: Scalar) {
    PURPLE(Scalar(255.0, 0.0, 255.0)),
    BLUE(Scalar(255.0, 0.0, 0.0)),
    GREEN(Scalar(0.0, 255.0, 0.0)),
    RED(Scalar(0.0, 0.0, 255.0)),
}

private class DigitClassifier {
    private val knn: KNearest = KNearest.create()
    private val numberOfNeighbours = 5

    init {
        val trainingLabeledImages = Files.walk(Path.of("src/main/resources/images/numbers"))
            .filter { it.extension == "png" }
            .map { Pair(it.parent.fileName.toString().toInt(), it) }
            .toList()
            .shuffled()
        val (trainingFeatures, trainingLabels) = trainingData(trainingLabeledImages)
        knn.defaultK = numberOfNeighbours
        knn.train(trainingFeatures, ROW_SAMPLE, trainingLabels)
    }

    fun classify(images: List<Mat>) = images.map { classify(it) }

    private fun classify(image: Mat): String {
        val neighbours = Mat()
        val champion = knn.findNearest(image.resize(20.0, 20.0).hogFeatures(), numberOfNeighbours, Mat(), neighbours)
            .toInt()
        return if (neighbours.areAll(champion))
            champion.toString()
        else
            "$champion?"
    }

    private fun trainingData(labeledImages: List<Pair<Int, Path>>): Triple<Mat, Mat, List<Path>> {
        val images = labeledImages.asSequence()
            .map { it.second }
            .map { loadImage(it.toString()) }
            .map { it.resize(20.0, 20.0) }
            .map { it.toGrayScale() }
            .toList()
        val features = images.toFeatures()
        val labels = vector_int_to_Mat(
            labeledImages
                .map { it.first }
                .toList()
        ).toFloats()
        return Triple(features, labels, labeledImages.map { it.second })
    }

    private fun List<Mat>.toFeatures(): Mat {
        val result = Mat()
        this
            .map { it.hogFeatures() }
            .forEach { result.push_back(it) }
        return result
    }

    private fun Mat.hogFeatures(): Mat {
        val hog = HOGDescriptor(
            Size(20.0, 20.0),
            Size(10.0, 10.0),
            Size(5.0, 5.0),
            Size(5.0, 5.0),
            9
        )

        val descriptor = MatOfFloat()
        hog.compute(this, descriptor)
        return descriptor.columnToRow()
    }

    private fun Mat.columnToRow(): Mat {
        val result = Mat(1, rows(), type())
        for (i in 0 until rows()) result.put(0, i, get(i, 0)[0])
        return result
    }

    private fun Mat.toFloats(): Mat {
        val destination = Mat(rows(), cols(), CV_32F)
        convertTo(destination, CV_32F)
        return destination
    }
}

private fun Mat.areAll(champion: Int): Boolean {
    for (i in 0 until rows())
        for (j in 0 until cols())
            if (get(i, j)[0].toInt() != champion)
                return false
    return true
}

