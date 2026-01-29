import SwiftUI

@main
struct BenchmarkAppApp: App {
    @StateObject private var benchmarkServer = BenchmarkServer()

    var body: some Scene {
        WindowGroup {
            MainTabView()
                .environmentObject(benchmarkServer)
                .onAppear {
                    benchmarkServer.start()
                }
        }
    }
}

struct MainTabView: View {
    var body: some View {
        TabView {
            CameraDetectionView()
                .tabItem {
                    Label("Camera", systemImage: "camera.fill")
                }

            ContentView()
                .tabItem {
                    Label("Benchmark", systemImage: "gauge.with.dots.needle.bottom.50percent")
                }
        }
    }
}
