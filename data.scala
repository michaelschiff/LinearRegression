import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList
processData.main()

object processData {
  def main() = {
    var stopWords = List("to", "a", "of", "and", "the", "him", "her", "they")
    println("loading data...")
    var tokens: IMat = load("/scratch/HW2/tokenized.mat", "tokens")
    tokens = tokens.t(?, 2)
    val words: CSMat = load("/scratch/HW2/tokenized.mat", "smap")
    println("data loaded")
    
    val ratingConverter:Map[String, Float] = Map("1" -> 1.0f,
                                                 "2" -> 2.0f,
                                                 "3" -> 3.0f,
                                                 "4" -> 4.0f,
                                                 "5" -> 5.0f)
    
    var reviewTextFlag = false
    var ratingFlag = false
    var reviewNumber = 0
    
    var currentXBlockRows:MutableList[Int] = MutableList[Int]()
    var currentXBlockCols:MutableList[Int] = MutableList[Int]()
    var currentXBlockVals:MutableList[Float] = MutableList[Float]()

    var currentYBlockRows:MutableList[Int] = MutableList[Int]()
    var currentYBlockCols:MutableList[Int] = MutableList[Int]()
    var currentYBlockVals:MutableList[Float] = MutableList[Float]()

    var currentReviewWordCounts:Map[Int,Float] = Map[Int,Float]()

    for ( iter:Int <- 0 to tokens.nrows-1 ) {
      val tokenIndex:Int = tokens(iter,0)-1 //indexes are 1 based in tokens
      if ( reviewTextFlag && tokenIndex < 100000 && !stopWords contains words(tokenIndex)) {
        if ( currentReviewWordCounts contains tokenIndex ) {
          currentReviewWordCounts(tokenIndex) += 1.0f
        } else {
          currentReviewWordCounts(tokenIndex) = 1.0f
        }
      }
      if ( ratingFlag ) {
        //add to y's reviewNumber is row, column is 0, value is label
        currentYBlockRows += reviewNumber % 10000
        currentYBlockCols += 0
        currentYBlockVals += ratingConverter(words(tokenIndex))
        ratingFlag = false
      }
      if ( words(tokenIndex) == "<rating>" ) { ratingFlag = true }
      if ( words(tokenIndex) == "<review_text>" ) { reviewTextFlag = true }
      if ( words(tokenIndex) == "</review_text>" ) { reviewTextFlag = false }
      if ( words(tokenIndex) == "</review>" ) {
        val totalNumWords:Float = currentReviewWordCounts.values.reduceLeft(_+_)
        currentReviewWordCounts.foreach( kv => currentReviewWordCounts(kv._1) = kv._2 / totalNumWords )
        for ( kv <- currentReviewWordCounts.toList ) {
          currentXBlockRows += kv._1
          currentXBlockCols += reviewNumber % 10000
          currentXBlockVals += kv._2
        }
        currentReviewWordCounts = Map[Int,Float]() //this collection is reset for EVERY REVIEW
        reviewNumber += 1
        if ( reviewNumber % 10000 == 0 ) {
          println("Saving Block Number: " + (reviewNumber/10000))
          saveAs("mats/XY"+(reviewNumber/10000),
                 sparse(icol(currentXBlockRows.toList), icol(currentXBlockCols.toList), col(currentXBlockVals.toList), 100000, 10000),
                 "X",
                 sparse(icol(currentYBlockRows.toList), icol(currentYBlockCols.toList), col(currentYBlockVals.toList), 10000, 1),
                 "Y")
          currentXBlockRows = MutableList[Int]()
          currentXBlockCols = MutableList[Int]()
          currentXBlockVals = MutableList[Float]()
          currentYBlockRows = MutableList[Int]()
          currentYBlockCols = MutableList[Int]()
          currentYBlockVals = MutableList[Float]()
        }
      }
    }
  }
}
