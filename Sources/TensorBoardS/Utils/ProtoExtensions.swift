extension TensorBoardS_TensorShapeProto.Dim: ExpressibleByIntegerLiteral {
    init(integerLiteral value: Int64) {
        self.init(size: value)
    }
    
    init(size: Int64) {
        self.init()
        self.size = size
    }
}

extension TensorBoardS_Summary.Image {
    init(png: PNGData) {
        self.init()
        self.width = Int32(png.width)
        self.height = Int32(png.height)
        self.colorspace = Int32(png.colorspace.rawValue)
        self.encodedImageString = png.data
    }
}
