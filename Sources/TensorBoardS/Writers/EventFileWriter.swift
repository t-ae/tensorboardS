import Foundation

class EventFileWriter {
    let logdir: URL
    let writer: EventsWriter
    
    let writerQueue: DispatchQueue
    
    var eventQueue: [TensorBoardS_Event] = []
    var flushTimer: Timer?
    
    private let lock = NSLock()
    
    init(
        logdir: URL,
        filenameSuffix: String,
        flushInterval: TimeInterval = 120
    ) throws {
        self.logdir = logdir
        
        writer = try EventsWriter(logdir: logdir,
                                  filenamePrefix: "events",
                                  filenameSuffix: filenameSuffix)
        
        writerQueue = DispatchQueue(label: "EventFileWriter")
        
        flushTimer = Timer.scheduledTimer(withTimeInterval: flushInterval, repeats: true) { _ in
            self.flush()
        }
    }
    
    func addEvent(event: TensorBoardS_Event) {
        lock.lock()
        defer { lock.unlock() }
        eventQueue.append(event)
    }
    
    func flush() {
        lock.lock()
        defer { lock.unlock() }
        
        while !eventQueue.isEmpty {
            let event = eventQueue.remove(at: 0)
            writer.write(event: event)
        }
        
        writer.flush()
    }
    
    func close() {
        flushTimer?.invalidate()
        flush()
        writer.close()
    }
    
    deinit {
        close()
    }
}
