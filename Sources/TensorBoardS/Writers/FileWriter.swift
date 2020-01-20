import Foundation

class FileWriter {
    let writer: EventFileWriter
    
    init(
        logdir: URL,
        flushInterval: TimeInterval = 120,
        filenameSuffix: String = ""
    ) throws {
        writer = try EventFileWriter(logdir: logdir,
                                     filenameSuffix: filenameSuffix,
                                     flushInterval: flushInterval)
    }
    
    func addEvent(_ event: TensorBoardS_Event, step: Int? = nil, date: Date? = nil) {
        var event = event
        if let step = step.map(Int64.init) {
            event.step = step
        }
        event.wallTime = (date ?? Date()).timeIntervalSince1970
        
        writer.addEvent(event: event)
    }
    
    func addSummary(_ summary: TensorBoardS_Summary, step: Int? = nil, date: Date? = nil) {
        let event = TensorBoardS_Event.with {
            $0.summary = summary
        }
        addEvent(event, step: step, date: date)
    }
    
    func flush() {
        writer.flush()
    }
    
    func close() {
        writer.close()
    }
}

