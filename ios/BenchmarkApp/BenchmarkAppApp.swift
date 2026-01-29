import SwiftUI

@main
struct BenchmarkAppApp: App {
    @StateObject private var benchmarkServer = BenchmarkServer()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(benchmarkServer)
                .onAppear {
                    benchmarkServer.start()
                }
        }
    }
}
