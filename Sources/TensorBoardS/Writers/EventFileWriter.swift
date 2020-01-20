import Foundation

class EventFileWriter {
    let writer: EventsWriter
    
    let writerQueue: DispatchQueue
    
    var eventQueue: [TensorBoardS_Event] = []
    var timerQueue = DispatchQueue.global()
    
    private var closed = false
    
    private let lock = NSLock()
    
    init(
        logdir: URL,
        filenameSuffix: String,
        flushInterval: TimeInterval = 120
    ) throws {
        writer = try EventsWriter(logdir: logdir,
                                  filenamePrefix: "events",
                                  filenameSuffix: filenameSuffix)
        
        writerQueue = DispatchQueue(label: "EventFileWriter")
        
        timerQueue.async {
            while true {
                self.lock.lock()
                let closed = self.closed
                if closed {
                    self.lock.unlock()
                    break
                }
                self.flush(withLock: false)
                self.lock.unlock()
                Thread.sleep(forTimeInterval: flushInterval)
            }
        }
    }
    
    func addEvent(event: TensorBoardS_Event) {
        lock.lock()
        defer { lock.unlock() }
        
        if closed {
            return
        }
        eventQueue.append(event)
    }
    
    func flush(withLock: Bool = true) {
        if withLock {
            lock.lock()
        }
        
        while !eventQueue.isEmpty {
            let event = eventQueue.remove(at: 0)
            writer.write(event: event)
        }
        
        writer.flush()
        
        if withLock {
            lock.unlock()
        }
    }
    
    func close() {
        guard !closed else {
            return
        }
        lock.lock()
        closed = true
        flush(withLock: false)
        writer.close()
        lock.unlock()
    }
    
    deinit {
        close()
    }
}
