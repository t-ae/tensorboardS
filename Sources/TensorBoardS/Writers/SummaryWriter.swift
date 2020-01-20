import Foundation
import TensorFlow

public class SummaryWriter {
    public let logdir: URL
    
    public var errorHandler: (Error)->Void = { _ in }
    
    private let writer: FileWriter
    
    public init(
        logdir: URL,
        flushInterval: TimeInterval = 120,
        filenameSuffix: String = ""
    ) throws {
        self.logdir = logdir
        
        writer = try FileWriter(logdir: logdir,
                                flushInterval: flushInterval,
                                filenameSuffix: filenameSuffix)
    }
    
    func wrap(_ function: () throws -> Void) {
        do {
            try function()
        } catch {
            errorHandler(error)
        }
    }
    
    /// Add scalar data to summary.
    public func addScalar(
        tag: String,
        scalar: Float,
        step: Int = 0,
        date: Date = Date()
    ) {
        let summary = Summaries.scalar(name: tag, scalar: scalar)
        writer.addSummary(summary, step: step, date: date)
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
            let summary = try Summaries.image(tag: tag, image: image)
            writer.addSummary(summary)
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
        let uint8 = Tensor<UInt8>(image.clipped(min: 0, max: 1) * 255)
        addImage(tag: tag, image: uint8, step: step, date: date)
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
        let image = makeGridImage(images: images, colSize: colSize, paddingValue: 0)
        addImage(tag: tag, image: image, step: step, date: date)
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
        let image = makeGridImage(images: images, colSize: colSize, paddingValue: 0)
        addImage(tag: tag, image: image, step: step, date: date)
    }
    
    /// Add histogram to summary.
//    public func addHistogram<Scalar: TensorFlowNumeric>(
//        tag: String,
//        values: Tensor<Scalar>,
//        step: Int = 0,
//        date: Date = Date()
//    ) {
//        let summary = Summaries.histogram(tag: tag, values: values)
//        writer.addSummary(summary)
//    }
    
    /// Add text data to summary.
    public func addText(
        tag: String,
        text: String,
        step: Int = 0,
        date: Date = Date()
    ) {
        wrap {
            let summary = try Summaries.text(tag: tag, text: text)
            writer.addSummary(summary, step: step, date: date)
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
            let jsonData = try encoder.encode(encodable)
            guard let encoded = String(data: jsonData, encoding: .utf8) else {
                throw GenericError("Failed to encode jsonData to String: jsonData=\(jsonData)")
            }
            let text = """
            <pre>
            \(encoded)
            </pre>
            """
            addText(tag: tag, text: text, step: step, date: date)
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
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        addJSONText(tag: tag, encodable: encodable, encoder: encoder, step: step, date: date)
    }
    
    public func flush() {
        writer.flush()
    }
    
    public func close() {
        flush()
        writer.close()
    }
}
