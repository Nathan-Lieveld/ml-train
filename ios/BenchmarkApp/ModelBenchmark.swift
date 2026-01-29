import Foundation
import CoreML

struct BenchmarkResult: Codable {
    let meanLatencyMs: Double
    let stdLatencyMs: Double
    let minLatencyMs: Double
    let maxLatencyMs: Double
    let iterations: Int
}

enum BenchmarkError: LocalizedError {
    case modelNotFound(String)
    case loadFailed(String)
    case predictionFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let path): return "Model not found: \(path)"
        case .loadFailed(let msg): return "Failed to load model: \(msg)"
        case .predictionFailed(let msg): return "Prediction failed: \(msg)"
        }
    }
}

struct ModelBenchmark {
    static func run(modelPath: String, iterations: Int) async throws -> BenchmarkResult {
        let modelURL: URL
        if modelPath.hasPrefix("/") {
            modelURL = URL(fileURLWithPath: modelPath)
        } else if let bundleURL = Bundle.main.url(forResource: modelPath, withExtension: "mlpackage") {
            modelURL = bundleURL
        } else {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            modelURL = docs.appendingPathComponent(modelPath)
        }

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw BenchmarkError.modelNotFound(modelURL.path)
        }

        let compiledURL = try await MLModel.compileModel(at: modelURL)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        guard let inputDesc = model.modelDescription.inputDescriptionsByName.values.first,
              let constraint = inputDesc.multiArrayConstraint else {
            throw BenchmarkError.loadFailed("Cannot determine input shape")
        }

        let shape = constraint.shape.map { $0.intValue }
        let inputArray = try MLMultiArray(shape: constraint.shape, dataType: .float16)
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [inputDesc.name: inputArray])

        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try model.prediction(from: inputFeatures)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            latencies.append(elapsed)
        }

        let mean = latencies.reduce(0, +) / Double(latencies.count)
        let variance = latencies.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(latencies.count)
        let std = sqrt(variance)
        let minVal = latencies.min() ?? 0
        let maxVal = latencies.max() ?? 0

        try? FileManager.default.removeItem(at: compiledURL)

        return BenchmarkResult(
            meanLatencyMs: mean,
            stdLatencyMs: std,
            minLatencyMs: minVal,
            maxLatencyMs: maxVal,
            iterations: iterations
        )
    }

    static func runFromData(_ modelData: Data, iterations: Int) async throws -> BenchmarkResult {
        let tempDir = FileManager.default.temporaryDirectory
        let modelURL = tempDir.appendingPathComponent(UUID().uuidString + ".mlpackage")
        try modelData.write(to: modelURL)
        defer { try? FileManager.default.removeItem(at: modelURL) }
        return try await run(modelPath: modelURL.path, iterations: iterations)
    }
}
