import Foundation
import TensorFlow
import CLodePNG

struct PNGData {
    enum ColorSpace: Int {
        case grayscale = 1
        case grayscaleAlpha = 2
        case rgb = 3
        case rgba = 4
        case digitalYUV = 5
        case bgra = 6
        // Same as TensorBoardS_Summary.Image.colorspace
        ///   1 - grayscale
        ///   2 - grayscale + alpha
        ///   3 - RGB
        ///   4 - RGBA
        ///   5 - DIGITAL_YUV
        ///   6 - BGRA
        
        static func `default`(for channels: Int) -> ColorSpace {
            switch channels {
            case 1:
                return .grayscale
            case 2:
                return .grayscaleAlpha
            case 3:
                return .rgb
            case 4:
                return .rgba
            default:
                fatalError("Invalid channel num: \(channels)")
            }
        }
        
        var colorType: LodePNGColorType {
            switch self {
            case .grayscale:
                return LCT_GREY
            case .grayscaleAlpha:
                return LCT_GREY_ALPHA
            case .rgb:
                return LCT_RGB
            case .rgba:
                return LCT_RGBA
            default:
                fatalError("Invalid color space: \(self)")
            }
        }
    }

    let width: Int
    let height: Int
    let data: Data
    let colorspace: ColorSpace
    
    init(width: Int, height: Int, data: Data, colorspace: ColorSpace) {
        self.width = width
        self.height = height
        self.data = data
        self.colorspace = colorspace
    }
    
    init(from tensor: Tensor<UInt8>) throws {
        precondition(tensor.rank == 2 || tensor.rank == 3)
        let width = tensor.shape[1]
        let height = tensor.shape[0]
        let channels = tensor.rank == 3 ? tensor.shape[2] : 1
        let colorspace = ColorSpace.default(for: channels)
        
        var png: UnsafeMutablePointer<UInt8>? = nil
        var pngSize: Int = 0
        var state = LodePNGState()
        lodepng_state_init(&state)
        defer {
            free(png)
            lodepng_state_cleanup(&state)
        }
        let imageData = tensor.scalars
        let code = lodepng_encode_memory(&png, &pngSize, imageData,
                                         UInt32(width), UInt32(height),
                                         colorspace.colorType, 8)
        
        guard code == 0 else {
            throw LodePNGError(errorCode: code)
        }
        
        let data = Data(UnsafeMutableBufferPointer<UInt8>(start: png, count: pngSize))
        
        self.init(width: width, height: height, data: data, colorspace: colorspace)
    }
}

public struct LodePNGError: Error {
    public let errorCode: UInt32
    public let errorText: String
    
    init(errorCode: UInt32) {
        self.errorCode = errorCode
        self.errorText = String(cString: lodepng_error_text(errorCode)!)
    }
}
