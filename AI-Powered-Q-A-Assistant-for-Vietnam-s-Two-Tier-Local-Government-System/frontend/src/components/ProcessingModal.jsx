import { useState, useEffect, useRef } from "react";
import { uploadAPI } from "../api/client";

export default function ProcessingModal({ isOpen, onClose, file, docId }) {
    const [logs, setLogs] = useState([]);
    const [currentStep, setCurrentStep] = useState(0);
    const [isComplete, setIsComplete] = useState(false);
    const [hasError, setHasError] = useState(false);
    const [stats, setStats] = useState({ chunks: "—", vectors: "—", dim: "—" });
    const terminalEndRef = useRef(null);
    const eventSourceRef = useRef(null);
    const isCompleteRef = useRef(false);

    const steps = [
        "Tải Models",
        "OCR & Nhận diện",
        "Parse Văn Bản",
        "Chunking",
        "FAISS Index",
    ];

    useEffect(() => {
        if (!isOpen || !docId) {
            // Reset state when closed
            setLogs([]);
            setCurrentStep(0);
            setIsComplete(false);
            setHasError(false);
            isCompleteRef.current = false;
            setStats({ chunks: "—", vectors: "—", dim: "—" });

            // Close any existing SSE connection
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
            return;
        }

        isCompleteRef.current = false;

        // ── Connect to SSE stream ──
        const url = uploadAPI.getStreamUrl(docId);
        const es = new EventSource(url);
        eventSourceRef.current = es;

        es.addEventListener("log", (e) => {
            try {
                const data = JSON.parse(e.data);
                setLogs((prev) => [...prev, data.line]);
                if (data.step > 0) {
                    setCurrentStep(data.step);
                }
            } catch {
                // ignore parse errors
            }
        });

        es.addEventListener("complete", (e) => {
            try {
                const data = JSON.parse(e.data);
                isCompleteRef.current = true;
                setIsComplete(true);
                setCurrentStep(steps.length);
                setStats({
                    chunks: data.num_chunks ?? "—",
                    vectors: data.faiss_total ?? "—",
                    dim: data.embedding_dim ? `${data.embedding_dim}d` : "—",
                });
            } catch {
                setIsComplete(true);
            }
            es.close();
        });

        es.addEventListener("pipeline_error", (e) => {
            try {
                const data = JSON.parse(e.data);
                const errors = Array.isArray(data.errors) ? data.errors.filter(Boolean) : [];
                const message = errors.join("; ") || data.message || "Pipeline thất bại";
                setHasError(true);
                setLogs((prev) => [
                    ...prev,
                    `❌ LỖI: ${message}`,
                ]);
            } catch {
                setHasError(true);
                setLogs((prev) => [...prev, "❌ LỖI: Pipeline thất bại"]);
            }
            es.close();
        });

        // Browser-level onerror (network issues)
        es.onerror = () => {
            if (isCompleteRef.current || es.readyState === EventSource.CLOSED) {
                return;
            }
            setHasError(true);
            setLogs((prev) => [...prev, "❌ Mất kết nối tới server hoặc stream xử lý bị ngắt."]);
            es.close();
        };

        return () => {
            es.close();
            eventSourceRef.current = null;
        };
    }, [isOpen, docId]);

    useEffect(() => {
        // Auto scroll to bottom of terminal
        terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    if (!isOpen) return null;

    const isDone = isComplete || hasError;

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
            {/* Backdrop */}
            <div className="fixed inset-0 bg-black/70 backdrop-blur-md" onClick={isDone ? onClose : null}></div>

            {/* Modal Content */}
            <div className="relative w-full max-w-4xl rounded-2xl bg-[#1C1C21] shadow-2xl flex flex-col overflow-hidden border border-white/10"
                 onClick={(e) => e.stopPropagation()}>
                
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute right-4 top-4 z-10 rounded-full p-2 text-gray-500 hover:bg-white/10 hover:text-white transition-colors"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="h-5 w-5">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="p-8">
                    {/* Stepper */}
                    <div className="flex items-center justify-between mb-8 relative px-4">
                        {/* Connecting Line */}
                        <div className="absolute top-5 left-12 right-12 h-[2px] bg-[#2A2B32] -z-0"></div>
                        <div 
                            className="absolute top-5 left-12 h-[2px] bg-green-500 -z-0 transition-all duration-500"
                            style={{ width: `${(Math.min(currentStep, steps.length - 1)) / (steps.length - 1) * 100}%` }}
                        ></div>

                        {steps.map((step, idx) => {
                            const stepNum = idx + 1;
                            const isCompleted = currentStep >= stepNum || isComplete;
                            const isActive = currentStep === idx && !isComplete && !hasError;
                            return (
                                <div key={step} className="flex flex-col items-center z-10 gap-2">
                                    <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors duration-300 ${
                                        isCompleted ? 'bg-green-500/20 border-green-500 text-green-500' :
                                        isActive ? 'bg-[#2A2B32] border-green-500 text-white' :
                                        'bg-[#2A2B32] border-[#3F414F] text-gray-500'
                                    }`}>
                                        {isCompleted ? (
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                                                <path fillRule="evenodd" d="M19.916 4.626a.75.75 0 0 1 .208 1.04l-9 13.5a.75.75 0 0 1-1.154.114l-6-6a.75.75 0 0 1 1.06-1.06l5.353 5.353 8.493-12.74a.75.75 0 0 1 1.04-.207Z" clipRule="evenodd" />
                                            </svg>
                                        ) : isActive ? (
                                            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                                        ) : (
                                            <span className="text-xs font-bold">{stepNum}</span>
                                        )}
                                    </div>
                                    <span className={`text-xs font-medium ${isCompleted || isActive ? 'text-green-500' : 'text-gray-500'}`}>
                                        {step}
                                    </span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Stats Badges */}
                    <div className="flex gap-4 mb-6 px-2">
                        <div className="flex items-center gap-2 bg-[#25262B] px-3 py-1.5 rounded-lg border border-white/5">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 text-blue-400">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                            </svg>
                            <span className="text-xs font-mono text-gray-300">{file?.name || "—"}</span>
                        </div>
                        <div className="flex items-center gap-2 bg-[#25262B] px-3 py-1.5 rounded-lg border border-white/5">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 text-purple-400">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25A2.25 2.25 0 0 1 13.5 18v-2.25Z" />
                            </svg>
                            <span className="text-xs font-mono text-gray-300">{stats.chunks} chunks</span>
                        </div>
                        <div className="flex items-center gap-2 bg-[#25262B] px-3 py-1.5 rounded-lg border border-white/5">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 text-cyan-400">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
                            </svg>
                            <span className="text-xs font-mono text-gray-300">{stats.vectors} vectors ({stats.dim})</span>
                        </div>
                    </div>

                    {/* Terminal Window */}
                    <div className="bg-[#0D0D0F] border border-white/5 rounded-xl p-4 h-80 overflow-y-auto font-mono text-sm leading-6">
                        {logs.map((log, i) => {
                            let textColor = "text-gray-300";
                            if (log.includes("✅") || log.includes("✔")) textColor = "text-green-400";
                            if (log.includes("❌") || log.includes("ERROR") || log.includes("FAILED")) textColor = "text-red-400";
                            if (log.includes("STEP") || log.includes("PIPELINE COMPLETED") || log.includes("Step")) textColor = "text-blue-400 font-bold";
                            if (log.includes("⚠") || log.includes("WARNING")) textColor = "text-yellow-400";
                            
                            return (
                                <div key={i} className={`${textColor} break-all mb-1`}>
                                    {log}
                                </div>
                            );
                        })}
                        {!isDone && (
                            <div className="text-gray-500 animate-pulse mt-2">_</div>
                        )}
                        <div ref={terminalEndRef} />
                    </div>
                </div>

                {/* Footer */}
                <div className="border-t border-white/10 bg-[#1C1C21] p-4 flex items-center justify-between px-8">
                    <div className="flex items-center gap-2">
                        {isComplete ? (
                            <div className="flex items-center gap-2 text-green-500">
                                <div className="bg-green-500/20 rounded-full p-1">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                                        <path fillRule="evenodd" d="M19.916 4.626a.75.75 0 0 1 .208 1.04l-9 13.5a.75.75 0 0 1-1.154.114l-6-6a.75.75 0 0 1 1.06-1.06l5.353 5.353 8.493-12.74a.75.75 0 0 1 1.04-.207Z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <span className="font-semibold text-sm">Hoàn tất xử lý!</span>
                            </div>
                        ) : hasError ? (
                            <div className="flex items-center gap-2 text-red-400">
                                <div className="bg-red-500/20 rounded-full p-1">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                                        <path fillRule="evenodd" d="M5.47 5.47a.75.75 0 0 1 1.06 0L12 10.94l5.47-5.47a.75.75 0 1 1 1.06 1.06L13.06 12l5.47 5.47a.75.75 0 1 1-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 0 1-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <span className="font-semibold text-sm">Xử lý thất bại</span>
                            </div>
                        ) : (
                            <span className="text-gray-400 text-sm font-medium">Đang xử lý tài liệu...</span>
                        )}
                    </div>
                    
                    <button 
                        onClick={onClose}
                        disabled={!isDone}
                        className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                            isDone 
                                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                                : 'bg-[#2A2B32] text-gray-500 cursor-not-allowed'
                        }`}
                    >
                        Đóng
                    </button>
                </div>
            </div>
        </div>
    );
}
