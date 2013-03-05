import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.ArrayBuffer
//import scala.math._
import scala.util.Random
  
//run.woodshed()
run.main()

class classifier(xTraining:ArrayBuffer[SMat], yTraining:ArrayBuffer[SMat], xTest:ArrayBuffer[SMat], yTest:ArrayBuffer[SMat], THRESHOLD:Float) {
  var errPlot = semilogx()
  var gradPlot = semilogx()
  var numFeatures:Int = xTraining(0).nrows
  var WEIGHTS:FMat = zeros(numFeatures, 1)
  var ALPHA:Float = 0.0001f //0.0000000001f //good alpha for watching the woodshed test descend
  var LAMBDA:Float = 0.0f
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
    var gs = X * twice_diff //var gs = X.t Tmult(twice_diff, null)
    //gs = gs /@ X.ncols
    return gs
  } 
  def blockAvgError(X: SMat, Y:FMat): Float = { 
    val e = X.Tmult(WEIGHTS, null) - Y
    return sum(sqrt(e *@ e), 1)(0,0) / X.ncols
  }
  var iters:Int = 1
  while( iters-1 < 100 ) { //classifier trains forever right now, ill add in a threshold if it looks like its converging
    //ALPHA = ALPHA * (1.0f / iters.toFloat)
    var sumOfL1Gradients:Float = 0.0f
    for ( blockNum <- 0 to xTraining.size-1 ) {
      val X:SMat = xTraining(blockNum)
      val Y:FMat = full(yTraining(blockNum))
      val gradients:FMat = blockGradient(X, Y)
      WEIGHTS -= (gradients * ALPHA) + (LAMBDA * sign(WEIGHTS)) //additive term is for Lasso Reg.
      sumOfL1Gradients += L1Column(gradients)
    }
    val avgOfL1Gradients = sumOfL1Gradients / xTraining.size
    var fp = 0.0f; var tp = 0.0f; var fn = 0.0f; var tn = 0.0f; 
    var sumOfBlockAvgError:Float = 0.0f
    for ( blockNum <- 0 to xTest.size-1 ) {
      val X:SMat = xTest(blockNum)
      val Y:FMat = full(yTest(blockNum))
      sumOfBlockAvgError += blockAvgError(X, Y)
    }
    val avgOfSumOfBlockAvgError:Float = sumOfBlockAvgError / xTest.size
    errPlot.addPoint(0, iters, avgOfSumOfBlockAvgError, true)
    gradPlot.addPoint(0, iters, avgOfL1Gradients, true)
    println("Iteration: " + iters + " Error: " + avgOfSumOfBlockAvgError)
    iters += 1
  }
  def plotROCS():Float = {
    val p = plot()
    var points:List[Tuple2[Float,Float]] = List()
    for ( q <- 2 to 5 ) {
      var tp = 0.0f; var fp = 0.0f; var tn = 0.0f; var fn = 0.0f
      for ( blockNum <- 0 to xTest.size-1 ) {
        val X:SMat = xTest(blockNum)
        val Y:FMat = full(yTest(blockNum))
        //calculations for precision and recall
        val combo:FMat = X Tmult(WEIGHTS, null)
        val ourPos:FMat = combo >= q
        val yPos:FMat = Y >= q
        tp += sum(ourPos *@ yPos, 1)(0,0)
        tn += combo.nrows - sum( (ourPos + yPos) > 0, 1 )(0,0)
        fp += sum((ourPos - yPos) > 0, 1)(0,0)
        fn += sum((ourPos - yPos) < 0, 1)(0,0)
      }
      val sensitivity:Float = tp / ( tp + fn )
      val specificity:Float = tn / ( fp + tn )
      p.addPoint(0, 1-specificity, sensitivity, true)
      points = (1-specificity, sensitivity) :: points
    }
    var auc:Float = 0.0f
    for ( p <- 0 to 2 ) {
      val x1:Float = points(p)._1; val y1:Float = points(p)._2
      val x2:Float = points(p+1)._1; val y2:Float = points(p+1)._2
      auc += (y1 + y2)*(x2 - x1)*0.5f
    }
    return auc
  }
  def featureWeights():Tuple2[List[Tuple2[Int, Float]],List[Tuple2[Int,Float]]] = {
    var p = List[Tuple2[Int,Float]]()
    var n = List[Tuple2[Int,Float]]()
    var wcopy = FMat(WEIGHTS)
    for ( i <- 0 to 4 ) {
      var mostPosVal = Float.NegativeInfinity; var mostPosIndex = 0
      var mostNegVal = Float.PositiveInfinity; var mostNegIndex = 0
      for ( j <- 0 to wcopy.nrows-1 ) {
        val v = wcopy(j,0)
        if ( v > mostPosVal ) { mostPosVal = v; mostPosIndex = j }
        if ( v < mostNegVal ) { mostNegVal = v; mostNegIndex = j }
      }
      p = (mostPosIndex, mostPosVal) :: p
      n = (mostNegIndex, mostNegVal) :: n
      wcopy(mostPosIndex,0) = 0
      wcopy(mostNegIndex,0) = 0
    }
    return (p, n)
  }
}

object run {
  def main() = {
    val folds = 1
    println("loading data")
    var words: CSMat = load("/scratch/HW2/tokenized.mat", "smap")
    //words = words(0 to 100000, 0)
    val xTraining:ArrayBuffer[SMat] = new ArrayBuffer()
    val yTraining:ArrayBuffer[SMat] = new ArrayBuffer()
    for ( i <- 1 to 97 ) { //my data is broken up into 97 blocks, each block is 10K reviews
      xTraining += load("mats/XY"+i, "X")
      yTraining += load("mats/XY"+i, "Y")
    }
    
    //DO THIS WHOLE THING 10 times, collecting an AUC score fore each one, then average those together
    println("starting 10 fold cross validation")
    var AUCS = 0.0f
    for ( j <- 1 to folds ) { //Do this process 10 times
      //pull out 9 corresponding blocks of X and Y to act as hold out
      val xTest:ArrayBuffer[SMat] = new ArrayBuffer()
      val yTest:ArrayBuffer[SMat] = new ArrayBuffer()
      for ( i <- 1 to 9 ) {
        val rng = new Random(scala.compat.Platform.currentTime)
        val randomBlockNumber = rng.nextInt(xTraining.size)
        xTest += xTraining.remove(randomBlockNumber)
        yTest += yTraining.remove(randomBlockNumber)
      }
      //initialize and train classifier, retrieve evaluations
      val c = new classifier(xTraining, yTraining, xTest, yTest, 0.00001f)
      AUCS += c.plotROCS
      val fw = c.featureWeights
      println("Most Positive Features:")
      for ( f <- fw._1 ) {
        println(" " + words(f._1) + " " + f._2)
      }
      println("Most Negative Features:")
      for ( f <- fw._2 ) {
        println(" " + words(f._1) + " " + f._2)
      }
      println("finished fold " + j)
      xTraining ++= xTest
      yTraining ++= yTest
    }
    AUCS = AUCS / folds.toFloat
    println("Average AUC: " + AUCS)
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
    val c = new classifier(xTrain, yTrain, xTest, yTest, 0.0001f)
  }
}
