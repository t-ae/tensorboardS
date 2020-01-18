extension TensorBoardS_TensorShapeProto.Dim: ExpressibleByIntegerLiteral {
    init(integerLiteral value: Int64) {
        self.init(size: value)
    }
    
    init(size: Int64) {
        self.init()
        self.size = size
    }
}
