import Foundation

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
    
    public func addScalar(tag: String, scalar: Float, step: Int = 0, date: Date = Date()) {
        let summary = Summaries.scalar(name: tag, scalar: scalar)
        writer.addSummary(summary, step: step, date: date)
    }
    
    public func addText(tag: String, text: String, step: Int = 0, date: Date = Date()) {
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
