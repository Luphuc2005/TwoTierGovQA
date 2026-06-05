import { useRef, useState } from "react";

const MAX_FILE_SIZE = 25 * 1024 * 1024;

function isPdfFile(file) {
    return file?.type === "application/pdf" || file?.name?.toLowerCase().endsWith(".pdf");
}

function formatFileSize(bytes) {
    if (!bytes) return "0 KB";
    const units = ["B", "KB", "MB", "GB"];
    const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / Math.pow(1024, power);
    return `${value.toFixed(value >= 10 || power === 0 ? 0 : 1)} ${units[power]}`;
}

export default function UploadModal({ isOpen, onClose, onUpload, externalError }) {
    const fileInputRef = useRef(null);
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [error, setError] = useState("");

    // Show external error from parent (e.g. upload API failure)
    const displayError = externalError || error;

    if (!isOpen) return null;

    const resetState = () => {
        setIsDragging(false);
        setSelectedFile(null);
        setError("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const handleClose = () => {
        resetState();
        onClose?.();
    };

    const pickFile = (fileList) => {
        const file = Array.from(fileList || [])[0];
        if (!file) return;

        if (!isPdfFile(file)) {
            setSelectedFile(null);
            setError("Chỉ hỗ trợ tệp PDF.");
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            setSelectedFile(null);
            setError("Tệp PDF cần nhỏ hơn 25 MB.");
            return;
        }

        setSelectedFile(file);
        setError("");
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) {
            setIsDragging(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        pickFile(e.dataTransfer.files);
    };

    const handleFileChange = (e) => {
        pickFile(e.target.files);
    };

    const handleUpload = () => {
        if (!selectedFile) {
            setError("Vui lòng chọn một tệp PDF trước khi thêm.");
            return;
        }

        onUpload?.([selectedFile]);
        resetState();
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="fixed inset-0 bg-black/70 backdrop-blur-md"
                onClick={handleClose}
            />

            <div
                className="relative max-h-[calc(100vh-2rem)] w-full max-w-2xl overflow-y-auto rounded-2xl border border-white/10 bg-[#1b1c24] text-left shadow-2xl shadow-black/40"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-blue-500/15 to-transparent" />

                <button
                    onClick={handleClose}
                    className="absolute right-4 top-4 z-10 rounded-lg p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    aria-label="Đóng"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth={2}
                        stroke="currentColor"
                        className="h-5 w-5"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="relative px-6 pb-6 pt-7 sm:px-8 sm:pb-8">
                    <div className="mb-6 flex items-start gap-4 pr-10">
                        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-blue-300/20 bg-blue-500/15 text-blue-200">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth={1.7}
                                stroke="currentColor"
                                className="h-6 w-6"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5A3.375 3.375 0 0 0 10.125 2.25H7.5A2.25 2.25 0 0 0 5.25 4.5v15A2.25 2.25 0 0 0 7.5 21.75h9A2.25 2.25 0 0 0 18.75 19.5v-2.25"
                                />
                            </svg>
                        </div>
                        <div>
                            <p className="text-sm font-medium text-blue-300">Tài liệu pháp luật</p>
                            <h2 className="mt-1 text-2xl font-semibold tracking-normal text-white">
                                Thêm PDF vào chatbot
                            </h2>
                            <p className="mt-2 text-sm leading-6 text-gray-400">
                                Tải lên văn bản PDF để hệ thống xử lý và đưa vào kho tra cứu.
                            </p>
                        </div>
                    </div>

                    <div
                        className={`relative flex min-h-[300px] flex-col items-center justify-center rounded-2xl border border-dashed p-6 text-center transition-all ${
                            isDragging
                                ? "border-blue-300 bg-blue-500/15 shadow-inner shadow-blue-950/30"
                                : "border-white/15 bg-[#242633]/80 hover:border-blue-300/45 hover:bg-[#272a38]"
                        }`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="mb-5 flex h-20 w-20 items-center justify-center rounded-2xl border border-white/10 bg-[#171923] shadow-lg shadow-black/20">
                            <div className="relative h-12 w-10 rounded-lg border border-red-300/40 bg-red-500/10 text-red-200">
                                <div className="absolute right-0 top-0 h-3 w-3 rounded-bl-md border-b border-l border-red-300/40 bg-[#242633]" />
                                <span className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] font-bold tracking-normal">
                                    PDF
                                </span>
                            </div>
                        </div>

                        <p className="text-lg font-semibold text-white">
                            Kéo thả PDF vào đây
                        </p>
                        <p className="mt-2 max-w-sm text-sm leading-6 text-gray-400">
                            Hỗ trợ một tệp PDF, tối đa 25 MB.
                        </p>

                        <input
                            type="file"
                            hidden
                            ref={fileInputRef}
                            accept="application/pdf,.pdf"
                            onChange={handleFileChange}
                        />

                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="mt-6 inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-950/30 transition-colors hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                                strokeWidth={1.8}
                                stroke="currentColor"
                                className="h-4 w-4"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M12 16.5V9.75m0 0 3 3m-3-3-3 3M6.75 19.5h10.5A2.25 2.25 0 0 0 19.5 17.25V6.75A2.25 2.25 0 0 0 17.25 4.5H6.75A2.25 2.25 0 0 0 4.5 6.75v10.5A2.25 2.25 0 0 0 6.75 19.5Z"
                                />
                            </svg>
                            Chọn PDF
                        </button>
                    </div>

                    {displayError && (
                        <div className="mt-4 rounded-lg border border-red-400/25 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                            {displayError}
                        </div>
                    )}

                    {selectedFile && (
                        <div className="mt-4 flex items-center gap-3 rounded-xl border border-blue-300/20 bg-blue-500/10 p-4">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-500/15 text-xs font-bold text-red-200">
                                PDF
                            </div>
                            <div className="min-w-0 flex-1">
                                <p className="truncate text-sm font-semibold text-white">
                                    {selectedFile.name}
                                </p>
                                <p className="mt-0.5 text-xs text-gray-400">
                                    {formatFileSize(selectedFile.size)}
                                </p>
                            </div>
                            <button
                                type="button"
                                onClick={() => {
                                    setSelectedFile(null);
                                    setError("");
                                    if (fileInputRef.current) fileInputRef.current.value = "";
                                }}
                                className="rounded-lg p-2 text-gray-400 transition-colors hover:bg-white/10 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                                aria-label="Bỏ chọn PDF"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    strokeWidth={2}
                                    stroke="currentColor"
                                    className="h-4 w-4"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                    )}

                    <div className="mt-6 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
                        <button
                            type="button"
                            onClick={handleClose}
                            className="rounded-xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-medium text-gray-200 transition-colors hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            Hủy
                        </button>
                        <button
                            type="button"
                            onClick={handleUpload}
                            disabled={!selectedFile}
                            className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-950/30 transition-colors hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:cursor-not-allowed disabled:bg-[#343746] disabled:text-gray-500 disabled:shadow-none"
                        >
                            Thêm vào chatbot
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
