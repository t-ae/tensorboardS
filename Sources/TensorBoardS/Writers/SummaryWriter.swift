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
