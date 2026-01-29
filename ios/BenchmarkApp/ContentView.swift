import SwiftUI
import CoreML
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var server: BenchmarkServer
    @State private var modelPath: String = ""
    @State private var iterations: Int = 100
    @State private var results: BenchmarkResult?
    @State private var isRunning = false
    @State private var errorMessage: String?
    @State private var showFilePicker = false

    var body: some View {
        NavigationView {
            Form {
                Section("Model") {
                    TextField("Model path or bundle name", text: $modelPath)
                    Button("Select from Files") {
                        showFilePicker = true
                    }
                }

                Section("Settings") {
                    Stepper("Iterations: \(iterations)", value: $iterations, in: 10...1000, step: 10)
                }

                Section("Actions") {
                    Button(action: runBenchmark) {
                        if isRunning {
                            ProgressView()
                        } else {
                            Text("Run Benchmark")
                        }
                    }
                    .disabled(modelPath.isEmpty || isRunning)
                }

                if let error = errorMessage {
                    Section("Error") {
                        Text(error)
                            .foregroundColor(.red)
                    }
                }

                if let r = results {
                    Section("Results") {
                        LabeledContent("Mean Latency", value: String(format: "%.3f ms", r.meanLatencyMs))
                        LabeledContent("Std Dev", value: String(format: "%.3f ms", r.stdLatencyMs))
                        LabeledContent("Min", value: String(format: "%.3f ms", r.minLatencyMs))
                        LabeledContent("Max", value: String(format: "%.3f ms", r.maxLatencyMs))
                        LabeledContent("Iterations", value: "\(r.iterations)")
                    }
                }

                Section("Server") {
                    LabeledContent("Status", value: server.isRunning ? "Running" : "Stopped")
                    LabeledContent("Port", value: "8765")
                }
            }
            .navigationTitle("ML Benchmark")
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [UTType(filenameExtension: "mlpackage") ?? .folder],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let url = urls.first {
                        modelPath = url.path
                    }
                case .failure(let error):
                    errorMessage = error.localizedDescription
                }
            }
            .onOpenURL { url in
                handleDeepLink(url)
            }
        }
    }

    private func runBenchmark() {
        isRunning = true
        errorMessage = nil
        results = nil

        Task {
            do {
                let result = try await ModelBenchmark.run(modelPath: modelPath, iterations: iterations)
                await MainActor.run {
                    results = result
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isRunning = false
                }
            }
        }
    }

    private func handleDeepLink(_ url: URL) {
        guard url.scheme == "benchmark", url.host == "run" else { return }

        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        if let items = components?.queryItems {
            for item in items {
                switch item.name {
                case "model":
                    modelPath = item.value ?? ""
                case "iterations":
                    iterations = Int(item.value ?? "") ?? 100
                default:
                    break
                }
            }
        }

        if !modelPath.isEmpty {
            runBenchmark()
        }
    }
}
