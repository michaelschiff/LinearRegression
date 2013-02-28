import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import scala.collection.mutable.ArrayBuffer
import scala.math
import scala.util.Random
  
run.woodshed()
run.main()

class classifier(xTraining:ArrayBuffer[SMat], yTraining:ArrayBuffer[SMat], xTest:ArrayBuffer[SMat], yTest:ArrayBuffer[SMat]) {
  var numFeatures:Int = xTraining(0).nrows
  var WEIGHTS:FMat = zeros(numFeatures, 1)
  var ALPHA:Float = 1.0f
  var LAMBDA:Float = 0.1f
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
  def L2Column(x:FMat): Float = math.sqrt(sum(x*@x, 1)(0,0)).toFloat
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
  var iters:Int = 0
  while( true ) { //classifier trains forever right now, ill add in a threshold if it looks like its converging
    var sumOfL2Gradients:Float = 0.0f
    for ( blockNum <- 0 to xTraining.size-1 ) {
      val X:SMat = xTraining(blockNum)
      val Y:FMat = full(yTraining(blockNum))
      val gradients:FMat = blockGradient(X, Y)
      WEIGHTS -= (gradients * ALPHA) + (LAMBDA * sign(WEIGHTS)) //additive term is for Lasso Reg.
      sumOfL2Gradients += L2Column(gradients)
    }
    val avgOfL2Gradients:Float = sumOfL2Gradients / xTraining.size
    var sumOfBlockAvgError:Float = 0.0f
    for ( blockNum <- 0 to xTest.size-1 ) {
      val X:SMat = xTest(blockNum)
      val Y:FMat = full(yTest(blockNum))
      sumOfBlockAvgError += blockAvgError(X, Y)
    }
    val avgOfSumOfBlockAvgError:Float = sumOfBlockAvgError / xTest.size
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    println("Iteration: " + iters)
    println("Average of the error from each block: " + avgOfSumOfBlockAvgError)
    println("Average of the L2 of the gradients from each block: " + avgOfL2Gradients)
    println("====================================================================")
    iters += 1
  }
}
object run {
  def main() = {
    val xTraining:ArrayBuffer[SMat] = new ArrayBuffer()
    val yTraining:ArrayBuffer[SMat] = new ArrayBuffer()
    for ( i <- 1 to 97 ) { //my data is broken up into 97 blocks, each block is 10K reviews
      xTraining += load("mats/XY"+i, "X")
      yTraining += load("mats/XY"+i, "Y")
    }
    //xTraining += load("mats/XYLast", "X")
    //yTraining += load("mats/XYLast", "Y")

    //pull out 9 corresponding blocks of X and Y to act as hold out
    val xTest:ArrayBuffer[SMat] = new ArrayBuffer()
    val yTest:ArrayBuffer[SMat] = new ArrayBuffer()
    for ( i <- 1 to 9 ) {
      val rng = new Random()
      val randomBlockNumber = rng.nextInt(xTraining.size)
      xTest += xTraining.remove(randomBlockNumber)
      yTest += yTraining.remove(randomBlockNumber)
    }
    
    //initialize and train classifier
    val c = new classifier(xTraining, yTraining, xTest, yTest)
  }
  def woodshed() = {
    val xTrain:ArrayBuffer[SMat] = new ArrayBuffer(sparse(1 on 0 on 0), 
                                                   sparse(2 on 0 on 0),
                                                   sparse(3 on 0 on 0),
                                                   sparse(4 on 0 on 0),
                                                   sparse(5 on 0 on 0), 
                                                   sparse(6 on 0 on 0), 
                                                   sparse(7 on 0 on 0), 
                                                   sparse(8 on 0 on 0))
    val yTrain:ArrayBuffer[FMat] = new ArrayBuffer(col(1), col(2), col(3), col(4), col(5), col(6), col(7), col(8))
    val xTest:ArrayBuffer[SMat] = new ArrayBuffer(sparse(9 on 0 on 0), sparse(10 on 0 on 0))
    val yTest:ArrayBuffer[FMat] = new ArrayBuffer(col(9), col(10))
    val c = new classifier(xTrain, yTrain, xTest, yTest)
  }
}
