import AVFoundation
import Vision
import CoreML
import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var detections: [Detection] = []
    @Published var fps: Double = 0
    @Published var isDetecting = false
    @Published var permissionGranted = false
    @Published var debugLog: [String] = []
    @Published var confidenceThreshold: Float = 0.6

    private let maxLogLines = 50

    let session = AVCaptureSession()
    private var videoOutput: AVCaptureVideoDataOutput?
    private let processingQueue = DispatchQueue(label: "com.mltrain.camera", qos: .userInitiated)

    // Thread-safe request storage (accessed from processingQueue)
    private let requestLock = NSLock()
    private var _detectionRequest: VNCoreMLRequest?
    private var currentRequest: VNCoreMLRequest? {
        get { requestLock.withLock { _detectionRequest } }
        set { requestLock.withLock { _detectionRequest = newValue } }
    }

    // FPS tracking
    private var frameCount = 0
    private var lastFPSUpdate = Date()

    // Class labels (COCO 80 classes for YOLOv8)
    private let cocoLabels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    override init() {
        super.init()
        log("CameraManager initialized")
    }

    func log(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let entry = "[\(timestamp)] \(message)"
        print(entry)  // Also print to Xcode console
        DispatchQueue.main.async {
            self.debugLog.append(entry)
            if self.debugLog.count > self.maxLogLines {
                self.debugLog.removeFirst()
            }
        }
    }

    func checkPermissions() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        log("Camera permission status: \(status.rawValue)")
        switch status {
        case .authorized:
            permissionGranted = true
            setupCamera()
        case .notDetermined:
            log("Requesting camera permission...")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                self?.log("Permission response: \(granted)")
                DispatchQueue.main.async {
                    self?.permissionGranted = granted
                    if granted {
                        self?.setupCamera()
                    }
                }
            }
        default:
            log("Camera permission denied")
            permissionGranted = false
        }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("Failed to access camera")
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: processingQueue)
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        if session.canAddOutput(output) {
            session.addOutput(output)
            videoOutput = output

            if let connection = output.connection(with: .video) {
                // Use deprecated API for iOS 16 compatibility
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
        }

        session.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
    }

    func loadModel(path: String) async throws {
        log("Loading model: \(path)")
        let modelURL: URL

        // Try bundle first
        if let bundleURL = Bundle.main.url(forResource: path, withExtension: "mlmodelc") {
            log("Found compiled model in bundle: \(bundleURL.lastPathComponent)")
            modelURL = bundleURL
        } else if let bundleURL = Bundle.main.url(forResource: path, withExtension: "mlpackage") {
            log("Found mlpackage in bundle: \(bundleURL.lastPathComponent)")
            modelURL = bundleURL
        } else if path.hasPrefix("/") {
            log("Using absolute path: \(path)")
            modelURL = URL(fileURLWithPath: path)
        } else {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            modelURL = docs.appendingPathComponent(path)
            log("Using documents path: \(modelURL.path)")
        }

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            log("ERROR: Model file not found at \(modelURL.path)")
            throw BenchmarkError.modelNotFound(modelURL.path)
        }

        // Compile if needed
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            log("Model already compiled")
            compiledURL = modelURL
        } else {
            log("Compiling model...")
            compiledURL = try await MLModel.compileModel(at: modelURL)
            log("Compiled to: \(compiledURL.lastPathComponent)")
        }

        // Load model
        log("Loading MLModel with computeUnits=all...")
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: config)

        // Log model info
        let inputs = mlModel.modelDescription.inputDescriptionsByName
        let outputs = mlModel.modelDescription.outputDescriptionsByName
        log("Model inputs: \(inputs.keys.joined(separator: ", "))")
        log("Model outputs: \(outputs.keys.joined(separator: ", "))")
        for (name, desc) in outputs {
            if let constraint = desc.multiArrayConstraint {
                log("  \(name) shape: \(constraint.shape)")
            } else {
                log("  \(name) type: \(desc.type.rawValue)")
            }
        }

        let visionModel = try VNCoreMLModel(for: mlModel)
        log("VNCoreMLModel created successfully")

        // Create detection request
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            if let error = error {
                self?.log("Detection error: \(error.localizedDescription)")
            }
            self?.processDetections(request: request)
        }
        request.imageCropAndScaleOption = .scaleFill

        currentRequest = request
        log("Model ready for detection")

        await MainActor.run {
            self.isDetecting = true
        }
    }

    func stopDetection() {
        currentRequest = nil
        DispatchQueue.main.async {
            self.isDetecting = false
            self.detections = []
        }
    }

    private var lastResultLogTime = Date.distantPast
    private var framesSinceLog = 0

    private func processDetections(request: VNRequest) {
        guard let results = request.results else {
            log("No results from Vision request")
            return
        }

        framesSinceLog += 1
        let shouldLog = Date().timeIntervalSince(lastResultLogTime) > 2.0  // Log every 2 seconds

        var newDetections: [Detection] = []

        let threshold = confidenceThreshold

        // Handle different result types
        if let observations = results as? [VNRecognizedObjectObservation] {
            // Standard object detection (NMS built into model)
            if shouldLog {
                log("Output: VNRecognizedObjectObservation x\(observations.count), threshold: \(threshold)")
            }
            for observation in observations where observation.confidence > threshold {
                if let topLabel = observation.labels.first {
                    let colorIndex = abs(topLabel.identifier.hashValue) % Detection.colors.count
                    newDetections.append(Detection(
                        label: topLabel.identifier,
                        confidence: observation.confidence,
                        boundingBox: observation.boundingBox,
                        color: Detection.colors[colorIndex]
                    ))
                }
            }
        } else if let observations = results as? [VNCoreMLFeatureValueObservation] {
            // YOLOv8 raw output - parse manually
            if shouldLog {
                log("Output: VNCoreMLFeatureValueObservation x\(observations.count)")
                for obs in observations {
                    if let arr = obs.featureValue.multiArrayValue {
                        log("  shape: \(arr.shape), type: \(arr.dataType.rawValue)")
                    }
                }
            }
            newDetections = parseYOLOOutput(observations)
        } else {
            if shouldLog {
                let types = results.map { String(describing: type(of: $0)) }
                log("Unknown result types: \(Set(types).joined(separator: ", "))")
            }
        }

        if shouldLog {
            log("Detections: \(newDetections.count), frames: \(framesSinceLog)")
            lastResultLogTime = Date()
            framesSinceLog = 0
        }

        DispatchQueue.main.async { [weak self] in
            self?.detections = newDetections
            self?.updateFPS()
        }
    }

    private func parseYOLOOutput(_ observations: [VNCoreMLFeatureValueObservation]) -> [Detection] {
        // YOLOv8 output format: [1, 84, 8400] where 84 = 4 (box) + 80 (classes)
        // Transposed: [1, 8400, 84]
        guard let observation = observations.first,
              let multiArray = observation.featureValue.multiArrayValue else {
            return []
        }

        var detections: [Detection] = []
        let threshold = confidenceThreshold
        let shape = multiArray.shape.map { $0.intValue }

        // Handle different output shapes
        if shape.count == 3 {
            let numDetections = shape[1]
            let numClasses = shape[2] - 4

            for i in 0..<min(numDetections, 8400) {
                // Get class scores and find max
                var maxScore: Float = 0
                var maxIdx = 0
                for c in 0..<numClasses {
                    let score = multiArray[[0, i, 4 + c] as [NSNumber]].floatValue
                    if score > maxScore {
                        maxScore = score
                        maxIdx = c
                    }
                }

                guard maxScore > threshold else { continue }

                // Get bounding box (center x, center y, width, height)
                let cx = multiArray[[0, i, 0] as [NSNumber]].floatValue / 640.0
                let cy = multiArray[[0, i, 1] as [NSNumber]].floatValue / 640.0
                let w = multiArray[[0, i, 2] as [NSNumber]].floatValue / 640.0
                let h = multiArray[[0, i, 3] as [NSNumber]].floatValue / 640.0

                let boundingBox = CGRect(
                    x: CGFloat(cx - w/2),
                    y: CGFloat(cy - h/2),
                    width: CGFloat(w),
                    height: CGFloat(h)
                )

                let label = maxIdx < cocoLabels.count ? cocoLabels[maxIdx] : "class_\(maxIdx)"
                let colorIndex = maxIdx % Detection.colors.count

                detections.append(Detection(
                    label: label,
                    confidence: maxScore,
                    boundingBox: boundingBox,
                    color: Detection.colors[colorIndex]
                ))
            }
        }

        // Apply simple NMS
        return nonMaxSuppression(detections, iouThreshold: 0.45)
    }

    private func nonMaxSuppression(_ detections: [Detection], iouThreshold: Float) -> [Detection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [Detection] = []

        for detection in sorted {
            var dominated = false
            for existing in kept {
                if iou(detection.boundingBox, existing.boundingBox) > iouThreshold {
                    dominated = true
                    break
                }
            }
            if !dominated {
                kept.append(detection)
            }
        }

        return Array(kept.prefix(20))  // Limit to top 20
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return Float(intersectionArea / unionArea)
    }

    private func updateFPS() {
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFPSUpdate)

        if elapsed >= 1.0 {
            fps = Double(frameCount) / elapsed
            frameCount = 0
            lastFPSUpdate = now
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let request = currentRequest,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}
