import Foundation
import Network

class BenchmarkServer: ObservableObject {
    @Published var isRunning = false
    private var listener: NWListener?
    private let port: UInt16 = 8765

    func start() {
        guard listener == nil else { return }

        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            listener?.stateUpdateHandler = { [weak self] state in
                DispatchQueue.main.async {
                    self?.isRunning = (state == .ready)
                }
            }
            listener?.newConnectionHandler = { [weak self] connection in
                self?.handleConnection(connection)
            }
            listener?.start(queue: .global(qos: .userInitiated))
        } catch {
            print("Failed to start server: \(error)")
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
        isRunning = false
    }

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInitiated))
        receiveData(connection: connection, accumulated: Data())
    }

    private func receiveData(connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            var newData = accumulated
            if let data = data { newData.append(data) }

            if isComplete || error != nil {
                self?.processRequest(connection: connection, data: newData)
            } else if let headerEnd = newData.range(of: Data("\r\n\r\n".utf8)) {
                let headerData = newData[..<headerEnd.lowerBound]
                let headerString = String(data: headerData, encoding: .utf8) ?? ""
                if let contentLength = self?.parseContentLength(headerString) {
                    let bodyStart = headerEnd.upperBound
                    let bodyReceived = newData.count - bodyStart
                    if bodyReceived >= contentLength {
                        self?.processRequest(connection: connection, data: newData)
                    } else {
                        self?.receiveData(connection: connection, accumulated: newData)
                    }
                } else {
                    self?.processRequest(connection: connection, data: newData)
                }
            } else {
                self?.receiveData(connection: connection, accumulated: newData)
            }
        }
    }

    private func parseContentLength(_ header: String) -> Int? {
        for line in header.components(separatedBy: "\r\n") {
            if line.lowercased().hasPrefix("content-length:") {
                let value = line.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)
                return Int(value)
            }
        }
        return nil
    }

    private func processRequest(connection: NWConnection, data: Data) {
        guard let headerEnd = data.range(of: Data("\r\n\r\n".utf8)) else {
            sendError(connection: connection, message: "Invalid HTTP request")
            return
        }

        let headerData = data[..<headerEnd.lowerBound]
        let bodyData = data[headerEnd.upperBound...]
        let headerString = String(data: headerData, encoding: .utf8) ?? ""
        let lines = headerString.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            sendError(connection: connection, message: "Missing request line")
            return
        }

        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else {
            sendError(connection: connection, message: "Invalid request line")
            return
        }

        let method = parts[0]
        let path = parts[1]

        if method == "POST" && path == "/benchmark" {
            handleBenchmark(connection: connection, body: Data(bodyData))
        } else {
            sendError(connection: connection, message: "Unknown endpoint", code: 404)
        }
    }

    private func handleBenchmark(connection: NWConnection, body: Data) {
        struct BenchmarkRequest: Decodable {
            let modelData: String
            let iterations: Int?
        }

        do {
            let request = try JSONDecoder().decode(BenchmarkRequest.self, from: body)
            guard let modelData = Data(base64Encoded: request.modelData) else {
                sendError(connection: connection, message: "Invalid base64 model data")
                return
            }

            let iterations = request.iterations ?? 100

            Task {
                do {
                    let result = try await ModelBenchmark.runFromData(modelData, iterations: iterations)
                    let responseData = try JSONEncoder().encode(result)
                    self.sendResponse(connection: connection, data: responseData)
                } catch {
                    self.sendError(connection: connection, message: error.localizedDescription)
                }
            }
        } catch {
            sendError(connection: connection, message: "Invalid JSON: \(error.localizedDescription)")
        }
    }

    private func sendResponse(connection: NWConnection, data: Data, code: Int = 200) {
        let status = code == 200 ? "OK" : "Error"
        let header = "HTTP/1.1 \(code) \(status)\r\nContent-Type: application/json\r\nContent-Length: \(data.count)\r\nConnection: close\r\n\r\n"
        var response = Data(header.utf8)
        response.append(data)
        connection.send(content: response, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private func sendError(connection: NWConnection, message: String, code: Int = 400) {
        let json = "{\"error\": \"\(message)\"}"
        sendResponse(connection: connection, data: Data(json.utf8), code: code)
    }
}
