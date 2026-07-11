/**
 * RetrievalPanel — slide-out panel showing retrieval pipeline results.
 * Displays ranked chunks with scores, metadata, and source references.
 */
import { useState } from "react";

export default function RetrievalPanel({ isOpen, onClose, chunks, timing, tier }) {
    const [expandedIdx, setExpandedIdx] = useState(null);

    if (!isOpen) return null;

    const toggleExpand = (idx) => {
        setExpandedIdx(expandedIdx === idx ? null : idx);
    };

    // Score color based on rerank score
    const getScoreColor = (score) => {
        if (score >= 6) return "text-emerald-600 bg-emerald-50 border-emerald-200";
        if (score >= 3) return "text-amber-600 bg-amber-50 border-amber-200";
        return "text-red-500 bg-red-50 border-red-200";
    };

    // Rank badge color
    const getRankColor = (rank) => {
        if (rank === 1) return "bg-gradient-to-br from-amber-400 to-amber-600 text-white";
        if (rank === 2) return "bg-gradient-to-br from-gray-300 to-gray-500 text-white";
        if (rank === 3) return "bg-gradient-to-br from-amber-600 to-amber-800 text-white";
        return "bg-gray-100 text-txt-secondary";
    };

    return (
        <>
            {/* Overlay */}
            <div
                className="fixed inset-0 bg-black/20 z-40 glass-overlay"
                onClick={onClose}
            />

            {/* Panel */}
            <div className="fixed right-0 top-0 h-screen w-[440px] max-w-[90vw] bg-white z-50 shadow-2xl flex flex-col animate-slide-in-right">
                {/* Header */}
                <div className="px-5 py-4 border-b border-bdr flex items-center justify-between flex-shrink-0">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-purple-700 flex items-center justify-center shadow-md">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-white">
                                <path fillRule="evenodd" d="M10.5 3.75a6.75 6.75 0 1 0 0 13.5 6.75 6.75 0 0 0 0-13.5ZM2.25 10.5a8.25 8.25 0 1 1 14.59 5.28l4.69 4.69a.75.75 0 1 1-1.06 1.06l-4.69-4.69A8.25 8.25 0 0 1 2.25 10.5Z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div>
                            <h2 className="text-[15px] font-bold text-txt-primary leading-tight">
                                Retrieval Pipeline
                            </h2>
                            <p className="text-xs text-txt-label leading-tight">
                                Kết quả tìm kiếm & xếp hạng
                            </p>
                        </div>
                    </div>

                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-gray-100 text-txt-muted hover:text-txt-primary transition-smooth"
                        aria-label="Đóng"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                        </svg>
                    </button>
                </div>

                {/* Stats bar */}
                <div className="px-5 py-3 bg-surface border-b border-bdr flex items-center gap-4 text-xs flex-shrink-0">
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-violet-500" />
                        <span className="text-txt-secondary font-medium">
                            {chunks?.length || 0} chunks
                        </span>
                    </div>
                    {timing && (
                        <div className="flex items-center gap-1.5">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5 text-txt-muted">
                                <path fillRule="evenodd" d="M1 8a7 7 0 1 1 14 0A7 7 0 0 1 1 8Zm7.75-4.25a.75.75 0 0 0-1.5 0V8c0 .414.336.75.75.75h3.25a.75.75 0 0 0 0-1.5h-2.5v-3.5Z" clipRule="evenodd" />
                            </svg>
                            <span className="text-txt-secondary">{timing.toFixed(0)}ms</span>
                        </div>
                    )}
                    {tier && (
                        <div className={`px-2 py-0.5 rounded-full font-semibold border ${
                            tier === "API" ? "bg-blue-50 text-blue-600 border-blue-200"
                            : tier === "LOCAL" ? "bg-emerald-50 text-emerald-600 border-emerald-200"
                            : "bg-gray-100 text-txt-muted border-bdr"
                        }`}>
                            {tier}
                        </div>
                    )}
                </div>

                {/* Chunk list */}
                <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
                    {(!chunks || chunks.length === 0) ? (
                        <div className="flex flex-col items-center justify-center h-full text-center px-6">
                            <div className="w-14 h-14 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-7 h-7 text-txt-muted">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
                                </svg>
                            </div>
                            <p className="text-txt-label text-sm font-medium mb-1">
                                Chưa có dữ liệu retrieval
                            </p>
                            <p className="text-txt-muted text-xs">
                                Gửi câu hỏi để xem kết quả tìm kiếm
                            </p>
                        </div>
                    ) : (
                        chunks.map((chunk, idx) => (
                            <div
                                key={idx}
                                className="border border-bdr rounded-xl bg-white shadow-card hover:shadow-card-lg transition-smooth overflow-hidden"
                            >
                                {/* Chunk header */}
                                <button
                                    onClick={() => toggleExpand(idx)}
                                    className="w-full px-4 py-3 flex items-start gap-3 text-left hover:bg-surface/50 transition-smooth"
                                >
                                    {/* Rank badge */}
                                    <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-bold shadow-sm ${getRankColor(chunk.rank)}`}>
                                        {chunk.rank}
                                    </div>

                                    <div className="flex-1 min-w-0">
                                        {/* Source reference */}
                                        <div className="flex items-center gap-2 flex-wrap mb-1">
                                            {chunk.van_ban && (
                                                <span className="text-xs font-semibold text-primary-700 bg-primary-50 px-2 py-0.5 rounded-md border border-primary-200 truncate max-w-[220px]">
                                                    {chunk.van_ban}
                                                </span>
                                            )}
                                            {chunk.dieu && (
                                                <span className="text-xs text-txt-secondary bg-gray-100 px-1.5 py-0.5 rounded">
                                                    Đ.{chunk.dieu}
                                                </span>
                                            )}
                                            {chunk.khoan && (
                                                <span className="text-xs text-txt-secondary bg-gray-100 px-1.5 py-0.5 rounded">
                                                    K.{chunk.khoan}
                                                </span>
                                            )}
                                            {chunk.diem && (
                                                <span className="text-xs text-txt-secondary bg-gray-100 px-1.5 py-0.5 rounded">
                                                    Đ.{chunk.diem}
                                                </span>
                                            )}
                                        </div>

                                        {/* Preview text */}
                                        <p className={`text-[13px] text-txt-secondary leading-relaxed ${expandedIdx === idx ? '' : 'line-clamp-2'}`}>
                                            {chunk.text}
                                        </p>
                                    </div>

                                    {/* Expand icon */}
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        viewBox="0 0 20 20"
                                        fill="currentColor"
                                        className={`w-4 h-4 text-txt-muted flex-shrink-0 mt-1 transition-transform ${expandedIdx === idx ? 'rotate-180' : ''}`}
                                    >
                                        <path fillRule="evenodd" d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" />
                                    </svg>
                                </button>

                                {/* Score bar — always visible */}
                                <div className="px-4 pb-3 flex items-center gap-3">
                                    <div className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] font-semibold border ${getScoreColor(chunk.score_rerank)}`}>
                                        <span>CE:</span>
                                        <span>{chunk.score_rerank.toFixed(2)}</span>
                                    </div>
                                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] font-medium text-txt-label bg-gray-50 border border-bdr">
                                        <span>Dense:</span>
                                        <span>{chunk.score_retrieval.toFixed(4)}</span>
                                    </div>
                                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] font-medium text-txt-muted bg-gray-50 border border-bdr">
                                        <span>ID:</span>
                                        <span>{chunk.chunk_id}</span>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Footer legend */}
                <div className="px-5 py-3 border-t border-bdr bg-surface flex-shrink-0">
                    <p className="text-[11px] text-txt-muted text-center">
                        CE = Cross-Encoder Rerank Score &bull; Dense = Bi-Encoder Cosine Similarity
                    </p>
                </div>
            </div>
        </>
    );
}
