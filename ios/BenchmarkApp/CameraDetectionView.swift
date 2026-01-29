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

    var body: some View {
        ZStack {
            CameraPreviewView(session: camera.session)
                .ignoresSafeArea()

            // Bounding boxes overlay
            DetectionOverlay(detections: camera.detections)

            // UI overlay
            VStack {
                // Top bar with FPS
                HStack {
                    Spacer()
                    FPSBadge(fps: camera.fps)
                        .padding()
                }

                Spacer()

                // Bottom controls
                VStack(spacing: 12) {
                    if let error = errorMessage {
                        Text(error)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.red.opacity(0.8))
                            .cornerRadius(8)
                    }

                    HStack {
                        Button(action: { showModelPicker = true }) {
                            Label(selectedModel.isEmpty ? "Select Model" : selectedModel,
                                  systemImage: "cube.box")
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }

                        if camera.isDetecting {
                            Button(action: { camera.stopDetection() }) {
                                Label("Stop", systemImage: "stop.fill")
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 10)
                                    .background(Color.red)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                    }
                }
                .padding()
                .background(Color.black.opacity(0.5))
            }
        }
        .sheet(isPresented: $showModelPicker) {
            ModelPickerView(selectedModel: $selectedModel) { modelPath in
                loadModel(path: modelPath)
            }
        }
        .onAppear {
            camera.checkPermissions()
            autoLoadBundledModel()
        }
    }

    private func autoLoadBundledModel() {
        guard !hasAutoLoaded else { return }
        hasAutoLoaded = true

        // Try to auto-load yolov8n if bundled
        let defaultModels = ["yolov8n", "yolov8s", "yolov8m"]
        for modelName in defaultModels {
            if Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") != nil ||
               Bundle.main.url(forResource: modelName, withExtension: "mlpackage") != nil {
                selectedModel = modelName
                loadModel(path: modelName)
                return
            }
        }
    }

    private func loadModel(path: String) {
        Task {
            do {
                try await camera.loadModel(path: path)
                await MainActor.run {
                    errorMessage = nil
                }
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

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        context.coordinator.previewLayer = previewLayer
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.previewLayer?.frame = uiView.bounds
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
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
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(4)
                        .background(detection.color)
                        .offset(y: -20)
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
        Text(String(format: "%.1f FPS", fps))
            .font(.system(.body, design: .monospaced))
            .fontWeight(.bold)
            .foregroundColor(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(color.opacity(0.8))
            .cornerRadius(8)
    }
}

// MARK: - Model Picker

struct ModelPickerView: View {
    @Binding var selectedModel: String
    @Environment(\.dismiss) var dismiss
    let onSelect: (String) -> Void

    @State private var customPath: String = ""
    @State private var bundledModels: [String] = []

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
                        Text("No bundled models found")
                            .foregroundColor(.secondary)
                    }
                }

                Section("Custom Path") {
                    TextField("Model path", text: $customPath)
                    Button("Load Custom Model") {
                        selectedModel = customPath
                        onSelect(customPath)
                        dismiss()
                    }
                    .disabled(customPath.isEmpty)
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
        if let resourcePath = Bundle.main.resourcePath {
            let fm = FileManager.default
            if let contents = try? fm.contentsOfDirectory(atPath: resourcePath) {
                for item in contents {
                    if item.hasSuffix(".mlmodelc") || item.hasSuffix(".mlpackage") {
                        let name = (item as NSString).deletingPathExtension
                        models.append(name)
                    }
                }
            }
        }
        bundledModels = models
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
