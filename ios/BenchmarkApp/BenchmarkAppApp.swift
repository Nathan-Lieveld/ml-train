import SwiftUI

@main
struct BenchmarkAppApp: App {
    @StateObject private var benchmarkServer = BenchmarkServer()

    var body: some Scene {
        WindowGroup {
            CameraDetectionView()
                .environmentObject(benchmarkServer)
                .onAppear {
                    benchmarkServer.start()
                }
        }
    }
}
