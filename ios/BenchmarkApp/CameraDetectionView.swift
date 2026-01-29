import SwiftUI
import AVFoundation
import Vision
import CoreML

struct CameraDetectionView: View {
    @StateObject private var camera = CameraManager()
    @State private var selectedModel: String = ""
    @State private var showModelPicker = false
    @State private var errorMessage: String?
    @State private var hasAutoLoaded = false
    @State private var showDebugLog = false

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Camera preview - full screen
                CameraPreviewView(session: camera.session)
                    .ignoresSafeArea()

                // Bounding boxes overlay
                DetectionOverlay(detections: camera.detections)
                    .ignoresSafeArea()

                // UI overlay
                VStack(spacing: 0) {
                    // Top bar with FPS and status
                    HStack {
                        // Status indicator
                        HStack(spacing: 4) {
                            Circle()
                                .fill(camera.permissionGranted ? Color.green : Color.red)
                                .frame(width: 8, height: 8)
                            Text(camera.isDetecting ? "Detecting" : "Camera")
                                .font(.caption)
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(12)

                        Spacer()

                        // Debug toggle
                        Button(action: { showDebugLog.toggle() }) {
                            Image(systemName: showDebugLog ? "ladybug.fill" : "ladybug")
                                .foregroundColor(.white)
                                .padding(6)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(8)
                        }

                        FPSBadge(fps: camera.fps)
                    }
                    .padding(.horizontal)
                    .padding(.top, geometry.safeAreaInsets.top + 8)

                    // Debug log panel
                    if showDebugLog {
                        DebugLogView(logs: camera.debugLog)
                            .frame(maxHeight: 200)
                            .padding(.horizontal, 8)
                    }

                    Spacer()

                    // Bottom controls
                    VStack(spacing: 8) {
                        if let error = errorMessage {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.red.opacity(0.8))
                                .cornerRadius(8)
                                .padding(.horizontal)
                        }

                        // Confidence threshold slider
                        HStack(spacing: 8) {
                            Text("Conf")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.8))
                            Slider(value: $camera.confidenceThreshold, in: 0.1...0.95, step: 0.05)
                                .tint(.blue)
                            Text(String(format: "%.0f%%", camera.confidenceThreshold * 100))
                                .font(.system(size: 12, weight: .medium, design: .monospaced))
                                .foregroundColor(.white)
                                .frame(width: 36)
                        }
                        .padding(.horizontal, 16)

                        HStack(spacing: 12) {
                            Button(action: { showModelPicker = true }) {
                                Label(selectedModel.isEmpty ? "Select Model" : selectedModel,
                                      systemImage: "cube.box")
                                    .font(.subheadline)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(8)
                            }

                            if camera.isDetecting {
                                Button(action: { camera.stopDetection() }) {
                                    Label("Stop", systemImage: "stop.fill")
                                        .font(.subheadline)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(Color.red)
                                        .foregroundColor(.white)
                                        .cornerRadius(8)
                                }
                            }
                        }
                    }
                    .padding()
                    .padding(.bottom, 60) // Space for tab bar
                    .background(
                        LinearGradient(
                            colors: [Color.black.opacity(0), Color.black.opacity(0.7)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
            }
        }
        .ignoresSafeArea()
        .sheet(isPresented: $showModelPicker) {
            ModelPickerView(selectedModel: $selectedModel) { modelPath in
                loadModel(path: modelPath)
            }
        }
        .onAppear {
            camera.checkPermissions()
            // Delay auto-load to let camera initialize
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                autoLoadBundledModel()
            }
        }
    }

    private func autoLoadBundledModel() {
        guard !hasAutoLoaded else { return }
        hasAutoLoaded = true

        camera.log("Searching for bundled models...")

        // List all bundle resources for debugging
        if let resourcePath = Bundle.main.resourcePath {
            let fm = FileManager.default
            if let contents = try? fm.contentsOfDirectory(atPath: resourcePath) {
                let models = contents.filter { $0.contains("yolo") || $0.hasSuffix(".mlmodelc") || $0.hasSuffix(".mlpackage") }
                camera.log("ML files in bundle: \(models.isEmpty ? "none" : models.joined(separator: ", "))")
            }
        }

        // Try to auto-load yolov8n if bundled
        let defaultModels = ["yolov8n", "yolov8s", "yolov8m"]
        for modelName in defaultModels {
            let hasCompiled = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") != nil
            let hasPackage = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") != nil
            if hasCompiled || hasPackage {
                camera.log("Found \(modelName) (compiled: \(hasCompiled), package: \(hasPackage))")
                selectedModel = modelName
                loadModel(path: modelName)
                return
            }
        }
        camera.log("No bundled models found")
    }

    private func loadModel(path: String) {
        errorMessage = nil
        Task {
            do {
                try await camera.loadModel(path: path)
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
}

// MARK: - Camera Preview

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.backgroundColor = .black
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // Layout is handled by PreviewView
    }

    class PreviewView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }

        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}

// MARK: - Detection Overlay

struct DetectionOverlay: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { geometry in
            ForEach(detections) { detection in
                let rect = detection.boundingBox.scaled(to: geometry.size)
                ZStack(alignment: .topLeading) {
                    Rectangle()
                        .stroke(detection.color, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)

                    Text("\(detection.label) \(Int(detection.confidence * 100))%")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(detection.color)
                        .offset(y: -16)
                }
                .position(x: rect.midX, y: rect.midY)
            }
        }
    }
}

// MARK: - FPS Badge

struct FPSBadge: View {
    let fps: Double

    var color: Color {
        if fps >= 30 { return .green }
        if fps >= 15 { return .yellow }
        return .red
    }

    var body: some View {
        Text(String(format: "%.0f", fps))
            .font(.system(size: 14, weight: .bold, design: .monospaced))
            .foregroundColor(.white)
            .frame(minWidth: 32)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(color.opacity(0.8))
            .cornerRadius(6)
    }
}

// MARK: - Model Picker

struct ModelPickerView: View {
    @Binding var selectedModel: String
    @Environment(\.dismiss) var dismiss
    let onSelect: (String) -> Void

    @State private var customPath: String = ""
    @State private var bundledModels: [String] = []
    @State private var allBundleFiles: [String] = []

    var body: some View {
        NavigationView {
            List {
                Section("Bundled Models") {
                    ForEach(bundledModels, id: \.self) { model in
                        Button(action: {
                            selectedModel = model
                            onSelect(model)
                            dismiss()
                        }) {
                            HStack {
                                Text(model)
                                Spacer()
                                if selectedModel == model {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .foregroundColor(.primary)
                    }

                    if bundledModels.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("No bundled models found")
                                .foregroundColor(.secondary)
                            Text("Build with model or load custom path")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                Section("Custom Path") {
                    TextField("Model name or path", text: $customPath)
                        .autocapitalization(.none)
                    Button("Load Model") {
                        selectedModel = customPath
                        onSelect(customPath)
                        dismiss()
                    }
                    .disabled(customPath.isEmpty)
                }

                // Debug section
                Section("Bundle Contents (Debug)") {
                    ForEach(allBundleFiles.prefix(20), id: \.self) { file in
                        Text(file)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    if allBundleFiles.count > 20 {
                        Text("... and \(allBundleFiles.count - 20) more")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear {
                findBundledModels()
            }
        }
    }

    private func findBundledModels() {
        var models: [String] = []
        var allFiles: [String] = []

        if let resourcePath = Bundle.main.resourcePath {
            let fm = FileManager.default
            if let contents = try? fm.contentsOfDirectory(atPath: resourcePath) {
                allFiles = contents.sorted()
                for item in contents {
                    if item.hasSuffix(".mlmodelc") || item.hasSuffix(".mlpackage") {
                        let name = (item as NSString).deletingPathExtension
                        models.append(name)
                    }
                }
            }
        }

        bundledModels = models
        allBundleFiles = allFiles
    }
}

// MARK: - Detection Model

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
    let color: Color

    static let colors: [Color] = [.red, .green, .blue, .orange, .purple, .cyan, .yellow, .pink]
}

extension CGRect {
    func scaled(to size: CGSize) -> CGRect {
        CGRect(
            x: minX * size.width,
            y: (1 - maxY) * size.height,
            width: width * size.width,
            height: height * size.height
        )
    }
}

// MARK: - Debug Log View

struct DebugLogView: View {
    let logs: [String]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(logs.enumerated()), id: \.offset) { index, log in
                        Text(log)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.green)
                            .id(index)
                    }
                }
                .padding(8)
            }
            .background(Color.black.opacity(0.85))
            .cornerRadius(8)
            .onChange(of: logs.count) { _ in
                if let last = logs.indices.last {
                    withAnimation {
                        proxy.scrollTo(last, anchor: .bottom)
                    }
                }
            }
        }
    }
}
