import Foundation
import TensorFlow

class ThrowingSummaryWriter {
    let logdir: URL
    
    let writer: FileWriter
    
    init(
        logdir: URL,
        flushInterval: TimeInterval = 120,
        filenameSuffix: String = ""
    ) throws {
        self.logdir = logdir
        
        writer = try FileWriter(logdir: logdir,
                                flushInterval: flushInterval,
                                filenameSuffix: filenameSuffix)
    }
    
    func addScalar<Scalar: BinaryInteger>(
        tag: String,
        scalar: Scalar,
        step: Int = 0,
        date: Date = Date()
    ) {
        let summary = Summaries.scalar(name: tag, scalar: Float(scalar))
        writer.addSummary(summary, step: step, date: date)
    }
    
    func addScalar<Scalar: BinaryFloatingPoint>(
        tag: String,
        scalar: Scalar,
        step: Int = 0,
        date: Date = Date()
    ) {
        let summary = Summaries.scalar(name: tag, scalar: Float(scalar))
        writer.addSummary(summary, step: step, date: date)
    }
    
    func addImage(
        tag: String,
        image: Tensor<UInt8>,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let summary = try Summaries.image(tag: tag, image: image)
        writer.addSummary(summary)
    }
    
    func addImage<Scalar: TensorFlowFloatingPoint>(
        tag: String,
        image: Tensor<Scalar>,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let uint8 = Tensor<UInt8>(image.clipped(min: 0, max: 1) * 255)
        try addImage(tag: tag, image: uint8, step: step, date: date)
    }
    
    func addImages(
        tag: String,
        images: Tensor<UInt8>,
        colSize: Int,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let image = makeGridImage(images: images, colSize: colSize, paddingValue: 0)
        try addImage(tag: tag, image: image, step: step, date: date)
    }
    
    func addImages<Scalar: TensorFlowFloatingPoint>(
        tag: String,
        images: Tensor<Scalar>,
        colSize: Int,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let image = makeGridImage(images: images, colSize: colSize, paddingValue: 0)
        try addImage(tag: tag, image: image, step: step, date: date)
    }
    
    /// Add histogram to summary.
//    func addHistogram<Scalar: TensorFlowNumeric>(
//        tag: String,
//        values: Tensor<Scalar>,
//        step: Int = 0,
//        date: Date = Date()
//    ) {
//        let summary = Summaries.histogram(tag: tag, values: values)
//        writer.addSummary(summary)
//    }
    
    func addText(
        tag: String,
        text: String,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let summary = try Summaries.text(tag: tag, text: text)
        writer.addSummary(summary, step: step, date: date)
    }
    
    func addJSONText<T: Encodable>(
        tag: String,
        encodable: T,
        encoder: JSONEncoder,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let jsonData = try encoder.encode(encodable)
        guard let encoded = String(data: jsonData, encoding: .utf8) else {
            throw GenericError("Failed to encode jsonData to String: jsonData=\(jsonData)")
        }
        let text = """
        <pre>
        \(encoded)
        </pre>
        """
        try addText(tag: tag, text: text, step: step, date: date)
    }
    
    func addJSONText<T: Encodable>(
        tag: String,
        encodable: T,
        step: Int = 0,
        date: Date = Date()
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        try addJSONText(tag: tag, encodable: encodable, encoder: encoder, step: step, date: date)
    }
    
    func flush() {
        writer.flush()
    }
    
    func close() {
        flush()
        writer.close()
    }
}

public class SummaryWriter {
    /// Takes 2 arguments, error and call stack symbols.
    public var errorHandler: (Error, [String])->Void = { error, callStack in
        print("[SummaryWriter] \(error)", stderr)
    }
    
    private let writer: ThrowingSummaryWriter
    
    public init(
        logdir: URL,
        flushInterval: TimeInterval = 120,
        filenameSuffix: String = ""
    ) throws {
        writer = try ThrowingSummaryWriter(
            logdir: logdir,
            flushInterval: flushInterval,
            filenameSuffix: filenameSuffix
        )
    }
    
    
    public var logdir: URL {
        writer.logdir
    }
    
    func wrap(_ function: () throws -> Void) {
        do {
            try function()
        } catch {
            let callStack = Thread.callStackSymbols
            errorHandler(error, callStack)
        }
    }
    
    /// Add scalar data to summary.
    public func addScalar<Scalar: BinaryInteger>(
        tag: String,
        scalar: Scalar,
        step: Int = 0,
        date: Date = Date()
    ) {
        writer.addScalar(tag: tag, scalar: scalar, step: step, date: date)
    }
    
    /// Add scalar data to summary.
    public func addScalar<Scalar: BinaryFloatingPoint>(
        tag: String,
        scalar: Scalar,
        step: Int = 0,
        date: Date = Date()
    ) {
        writer.addScalar(tag: tag, scalar: scalar, step: step, date: date)
    }
    
    /// Add image data to summary
    ///
    /// - Parameters:
    ///   - image: Tensor which has shape [height, width, channels]. Gray/RGB/RGBA are supported.
    public func addImage(
        tag: String,
        image: Tensor<UInt8>,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addImage(tag: tag, image: image, step: step, date: date)
        }
    }
    
    /// Add image data to summary.
    ///
    /// - Parameters:
    ///   - image: Tensor which has shape [height, width, channels]. Gray/RGB/RGBA are supported. Asuumes pixel values are in [0, 1] range.
    public func addImage<Scalar: TensorFlowFloatingPoint>(
        tag: String,
        image: Tensor<Scalar>,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addImage(tag: tag, image: image, step: step, date: date)
        }
    }
    
    /// Add images as grid image to summary.
    ///
    /// - Parameters:
    ///   - images: Tensor which has shape [N, height, width, channels]. Gray/RGB/RGBA are supported.
    ///   - colSize: Number of grid columns.
    public func addImages(
        tag: String,
        images: Tensor<UInt8>,
        colSize: Int,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addImages(tag: tag, images: images,colSize: colSize,step: step, date: date)
        }
    }
    
    /// Add images as grid image to summary.
    ///
    /// - Parameters:
    ///   - images: Tensor which has shape [N, height, width, channels]. Gray/RGB/RGBA are supported. Asuumes pixel values are in [0, 1] range.
    ///   - colSize: Number of grid columns.
    public func addImages<Scalar: TensorFlowFloatingPoint>(
        tag: String,
        images: Tensor<Scalar>,
        colSize: Int,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addImages(tag: tag, images: images,colSize: colSize,step: step, date: date)
        }
    }
    
    /// Add text data to summary.
    public func addText(
        tag: String,
        text: String,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addText(tag: tag, text: text, step: step, date: date)
        }
    }
    
    /// Add JSON data of encodable as text.
    ///
    /// - Parameters:
    ///   - encoder: JSONEncoder to be used in encoding.
    public func addJSONText<T: Encodable>(
        tag: String,
        encodable: T,
        encoder: JSONEncoder,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addJSONText(tag: tag,
                                   encodable: encodable,
                                   encoder: encoder,
                                   step: step,
                                   date: date)
        }
    }
    
    /// Add JSON data of encodable as text.
    ///
    /// JSON data will be pretty-printed.
    public func addJSONText<T: Encodable>(
        tag: String,
        encodable: T,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            try writer.addJSONText(tag: tag, encodable: encodable, step: step, date: date)
        }
    }
    
    public func flush() {
        writer.flush()
    }
    
    public func close() {
        flush()
        writer.close()
    }
}
