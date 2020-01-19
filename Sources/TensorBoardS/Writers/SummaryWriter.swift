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
        do {
            let summary = try Summaries.image(tag: tag, image: image)
            writer.addSummary(summary)
        } catch {
            errorHandler(error)
        }
    }
    
    /// Add image data to summary.
    ///
    /// - Parameters:
    ///   - image: Tensor which has shape [height, width, channels]. Gray/RGB/RGBA are supported. Asuumes pixel values are in [0, 1] range.
    public func addImage<F: TensorFlowFloatingPoint>(
        tag: String,
        image: Tensor<F>,
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
    public func addImages<F: TensorFlowFloatingPoint>(
        tag: String,
        images: Tensor<F>,
        colSize: Int,
        step: Int = 0,
        date: Date = Date()
    ) {
        let image = makeGridImage(images: images, colSize: colSize, paddingValue: 0)
        addImage(tag: tag, image: image, step: step, date: date)
    }
    
    /// Add text data to summary.
    public func addText(
        tag: String,
        text: String,
        step: Int = 0,
        date: Date = Date()
    ) {
        do {
            let text = try Summaries.text(tag: tag, text: text)
            writer.addSummary(text, step: step, date: date)
        } catch {
            errorHandler(error)
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
