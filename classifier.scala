import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.ArrayBuffer
import scala.math
import scala.util.Random
  
//run.woodshed()
run.main()

class classifier(xTraining:ArrayBuffer[SMat], yTraining:ArrayBuffer[SMat], xTest:ArrayBuffer[SMat], yTest:ArrayBuffer[SMat], THRESHOLD:Float) {
  var roc = plot()
  var myPlot = plot()
  var myPlot2 = plot()
  var numFeatures:Int = xTraining(0).nrows
  var WEIGHTS:FMat = zeros(numFeatures, 1)
  var ALPHA:Float = 0.1f //0.0000000001f good alpha for watching the woodshed test descend
  var LAMBDA:Float = 0.000000001f
  if ( xTraining.size != yTraining.size ) { println("# training examples and # training labels do not match") }
  if ( xTest.size != yTest.size ) { println("# test examples and # test labels do not match") }
  for ( i <- 0 to xTraining.size-1 ) {
    if ( xTraining(i).nrows != numFeatures ) { println(i +"th block has wrong number of features") }
    if ( xTraining(i).ncols != yTraining(i).nrows ) { println(i+"th block of X and Y dimension mismatch") }
  }
  for ( i <- 0 to xTest.size-1 ) {
    if ( xTest(i).nrows != numFeatures ) { println(i + "th block of test data has wrong number of features") }
    if ( xTest(i).ncols != yTest(i).nrows ) { println(i + "th block of test X and Y dimension mismatch") }
  }
  def sign(x:FMat): FMat = ((2 * (x >= 0) - 1)+(2 * (x > 0) -1))/@2
  def L1Column(x:FMat): Float = sum(abs(x), 1)(0,0)
  def blockGradient(X:SMat, Y:FMat):FMat = {
    if ( X.ncols != Y.nrows ) { println("ERROR: block dimensions to not match") }
    val combo = X Tmult(WEIGHTS, null) //X is sparse w is a COLUMN!!!
    val diff = combo - Y
    val twice_diff = diff * 2.0f
    var gs = X * twice_diff
    gs = gs /@ X.ncols
    return gs
  } 
  def blockAvgError(X: SMat, Y:FMat): Float = { 
    val e = X.Tmult(WEIGHTS, null) - Y
    return sum(sqrt(e *@ e), 1)(0,0) / X.ncols
  }
  var iters:Int = 1
  var avgOfL1Gradients:Float = 1.0f
  var fp:Float = 0.0f; var tp:Float = 0.0f; var fn:Float = 0.0f; var tn:Float = 0.0f; 
  while( avgOfL1Gradients > THRESHOLD ) { //classifier trains forever right now, ill add in a threshold if it looks like its converging
    //ALPHA = ALPHA * (1.0f / iters.toFloat)
    var sumOfL1Gradients:Float = 0.0f
    for ( blockNum <- 0 to xTraining.size-1 ) {
      val X:SMat = xTraining(blockNum)
      val Y:FMat = full(yTraining(blockNum))
      val gradients:FMat = blockGradient(X, Y)
      WEIGHTS -= (gradients * ALPHA) + (LAMBDA * sign(WEIGHTS)) //additive term is for Lasso Reg.
      sumOfL1Gradients += L1Column(gradients)
    }
    avgOfL1Gradients = sumOfL1Gradients / xTraining.size
    fp = 0.0f; tp = 0.0f; fn = 0.0f; tn = 0.0f; 
    var sumOfBlockAvgError:Float = 0.0f
    for ( blockNum <- 0 to xTest.size-1 ) {
      val X:SMat = xTest(blockNum)
      val Y:FMat = full(yTest(blockNum))
      sumOfBlockAvgError += blockAvgError(X, Y)
      //calculations for precision and recall
      val combo:FMat = X Tmult(WEIGHTS, null)
      val ourPos:FMat = combo >= 5
      val yPos:FMat = Y >= 5
      tp += sum(ourPos *@ yPos, 1)(0,0)
      tn += combo.nrows - sum( (ourPos + yPos) > 0, 1 )(0,0)
      fp += sum((ourPos - yPos) > 0, 1)(0,0)
      fn += sum((ourPos - yPos) < 0, 1)(0,0)
    }
    val avgOfSumOfBlockAvgError:Float = sumOfBlockAvgError / xTest.size
    if ( (tp+fp+tn+fn) != (10000*xTest.size) ) { println("Math Error") }
    val precision:Float = tp / (tp + fp)
    val recall:Float = tp / (tp + fn)
    val accuracy:Float = (tp + tn) / (tp + fp + tn + fn)
    val F1:Float = (2*precision*recall) / (precision + recall)
    val sensitivity:Float = tp / ( tp + fn )
    val specificity:Float = tn / ( fp + tn )
    roc.addPoint(0, 1-specificity, sensitivity)
    myPlot.addPoint(0, iters-1, sensitivity, true)
    myPlot.addPoint(1, iters-1, 1-specificity, true)
    myPlot.addPoint(2, iters-1, specificity, true)
    myPlot2.addPoint(0, iters-1, avgOfSumOfBlockAvgError, true)
    myPlot2.addPoint(1, iters-1, avgOfL1Gradients, true)
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    println("Iteration: " + iters)
    println("Average of the error from each block: " + avgOfSumOfBlockAvgError)
    println("Average of the L1 norm of the gradients from each block: " + avgOfL1Gradients)
    println("Precision: " + precision)
    println("Recall: " + accuracy)
    println("F1: " + F1)
    println("Accuracy: " + accuracy)
    println("Sensitivity (tpr): " + sensitivity)
    println("Specificity (tnr): " + specificity)
    println("1 - Specificity (fpr): " + (1-specificity))
    println("True Negatives: " + tn )
    println("Total Negatives: " + (tn + fp))
    println("True Positives: " + tp )
    println("Total Positives: " + (tp + fn))
    println("====================================================================")
    iters += 1
  }
  def getRates():Tuple4[Float, Float, Float, Float] = (tp, fp, tn, fn)
}
object run {
  def main() = {
    val xTraining:ArrayBuffer[SMat] = new ArrayBuffer()
    val yTraining:ArrayBuffer[SMat] = new ArrayBuffer()
    for ( i <- 1 to 97 ) { //my data is broken up into 97 blocks, each block is 10K reviews
      xTraining += load("mats/XY"+i, "X")
      yTraining += load("mats/XY"+i, "Y")
    }

    var tp:Float = 0.0f; var fp:Float= 0.0f; var tn:Float= 0.0f; var fn:Float = 0.0f
    for ( j <- 0 to 9 ) { 
      //pull out 9 corresponding blocks of X and Y to act as hold out
      val xTest:ArrayBuffer[SMat] = new ArrayBuffer()
      val yTest:ArrayBuffer[SMat] = new ArrayBuffer()
      for ( i <- 1 to 9 ) {
        val rng = new Random()
        val randomBlockNumber = rng.nextInt(xTraining.size)
        xTest += xTraining.remove(randomBlockNumber)
        yTest += yTraining.remove(randomBlockNumber)
      }
      //initialize and train classifier, retrieve evaluations
      val c = new classifier(xTraining, yTraining, xTest, yTest, 0.00001f)
      var rates = c.getRates()
      tp += rates._1; fp += rates._2; tn += rates._3; fn += rates._4
      //restore training examples
      xTraining ++ xTest
      yTraining ++ yTest
    }
    tp = tp / 10.0f; fp = fp / 10.0f; tn = tn / 10.0f; fn = fn / 10.0f

    //make a ROC plot with the averages
    
  }
  def woodshed() = {
    val xTrain:ArrayBuffer[SMat] = new ArrayBuffer()
    xTrain += sparse(1 on 0 on 0)
    xTrain += sparse(2 on 0 on 0)
    xTrain += sparse(3 on 0 on 0)
    xTrain += sparse(4 on 0 on 0)
    xTrain += sparse(5 on 0 on 0)
    xTrain += sparse(6 on 0 on 0)
    xTrain += sparse(7 on 0 on 0)
    xTrain += sparse(8 on 0 on 0)
    val yTrain:ArrayBuffer[SMat] = new ArrayBuffer()
    yTrain += sparse(col(1))
    yTrain += sparse(col(2))
    yTrain += sparse(col(3))
    yTrain += sparse(col(4))
    yTrain += sparse(col(5))
    yTrain += sparse(col(6))
    yTrain += sparse(col(7))
    yTrain += sparse(col(8))
    val xTest:ArrayBuffer[SMat] = new ArrayBuffer()
    xTest += sparse(9 on 0 on 0)
    xTest += sparse(10 on 0 on 0)
    val yTest:ArrayBuffer[SMat] = new ArrayBuffer()
    yTest += sparse(col(9))
    yTest += sparse(col(10))
    val c = new classifier(xTrain, yTrain, xTest, yTest, 0.0000001f)
  }
}
