/**
  * Created by peadarcoyle on 23/12/15.
  */
package scalable_ml

import org.apache.spark.SparkContext
import org.scalatest._

class LeastSquareRegressionTest extends FunSuite {
  test("Simple run of least-squares regression")
  {
    val sc = new SparkContext("local", "LeastSquareRegressionTest")

    val dataset = new LinearExampleDataset(100, 4, 0.1)
    val lds = sc.parallelize(dataset.labeledPoints)

    val lsr = new LeastSquaresRegression()

    val weights = lsr.fit(lds)

    println("Real weights = " + dataset.weights.toSeq)
    println("Fitted weights = " + weights)

    sc.stop()
  }
}
