import Foundation
import TensorFlow

func makeGridImage<Scalar>(images: Tensor<Scalar>, colSize: Int, paddingValue: Scalar) -> Tensor<Scalar> {
    
    let originalRank = images.rank
    var images = images
    switch originalRank {
    case 3: // grayscale and no channel axis
        images = images.expandingShape(at: 3)
    case 4:
        break
    default:
        fatalError("Invalid images shape: \(images.shape)")
    }
    
    let rowSize = Int(ceil(Float(images.shape[0]) / Float(colSize)))
    let paddingSize = rowSize*colSize - images.shape[0]
    
    let width = images.shape[2]
    let height = images.shape[1]
    let channels = images.shape[3]
    
    let paddingShape = TensorShape([paddingSize, height, width, channels])
    let padding = Tensor<Scalar>.init(repeating: paddingValue, shape: paddingShape)
    
    var grid = Tensor(concatenating: [images, padding], alongAxis: 0)
    grid = grid.reshaped(to: TensorShape([rowSize, colSize, height, width, channels]))
    grid = grid.transposed(permutation: [0, 2, 1, 3, 4])
    grid = grid.reshaped(to: [rowSize*height, colSize*width, channels])
    
    if originalRank == 3 {
        // Remove channel axis
        grid = grid.squeezingShape(at: 3)
    }
    
    return grid
}
