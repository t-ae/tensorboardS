import Foundation

class EventFileWriter {
    private let writer: EventsWriter
    
    private var eventQueue: [TensorBoardS_Event] = []
    private var timer: Timer? = nil
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
        
        DispatchQueue.global(qos: .background).async {
            let timer = Timer.scheduledTimer(withTimeInterval: flushInterval, repeats: true) { _ in
                self.flush()
            }
            let runLoop = RunLoop.current
            runLoop.add(timer, forMode: .default)
            runLoop.run()
            self.timer = timer
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
        timer?.invalidate()
        flush(withLock: false)
        writer.close()
        lock.unlock()
    }
    
    deinit {
        close()
    }
}
