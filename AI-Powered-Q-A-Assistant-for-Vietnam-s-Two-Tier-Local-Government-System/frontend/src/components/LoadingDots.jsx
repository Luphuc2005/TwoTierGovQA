/**
 * LoadingDots — AI typing indicator with status text.
 */
export default function LoadingDots() {
    return (
        <div className="flex gap-3 mb-5 animate-fade-in">
            {/* AI Avatar */}
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center flex-shrink-0 shadow-sm">
                <span className="text-xs font-bold text-white">AI</span>
            </div>

            {/* Loading bubble with text */}
            <div className="bg-white border border-bdr rounded-2xl rounded-tl-sm px-5 py-4 shadow-card">
                <div className="flex items-center gap-3">
                    <div className="loading-dots flex space-x-1.5">
                        <span className="w-2 h-2 bg-primary-500 rounded-full inline-block" />
                        <span className="w-2 h-2 bg-primary-500 rounded-full inline-block" />
                        <span className="w-2 h-2 bg-primary-500 rounded-full inline-block" />
                    </div>
                    <span className="text-sm text-txt-label">
                        AI đang phân tích văn bản...
                    </span>
                </div>
            </div>
        </div>
    );
}
