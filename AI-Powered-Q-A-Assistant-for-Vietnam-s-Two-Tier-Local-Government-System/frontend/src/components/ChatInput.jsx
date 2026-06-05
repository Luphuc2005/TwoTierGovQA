/**
 * ChatInput — polished input bar with clear styling,
 * prominent send button, and Vietnamese placeholder.
 */
import { useState } from "react";

export default function ChatInput({ onSend, disabled, onOpenUpload }) {
    const [message, setMessage] = useState("");

    const handleSubmit = (e) => {
        e.preventDefault();
        const trimmed = message.trim();
        if (!trimmed || disabled) return;
        onSend(trimmed);
        setMessage("");
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const hasContent = message.trim().length > 0;

    return (
        <div className="bg-white border-t border-bdr px-4 py-4">
            <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
                <div className="flex items-center gap-2.5">
                    {/* Upload button */}
                    <button
                        type="button"
                        onClick={onOpenUpload}
                        disabled={disabled}
                        id="btn-upload"
                        className="group flex h-11 w-11 items-center justify-center rounded-xl
                         border border-bdr-medium bg-white text-txt-label
                         shadow-card transition-smooth flex-shrink-0
                         hover:border-primary-400 hover:text-primary-600 hover:bg-primary-50
                         hover:shadow-card-lg
                         focus:outline-none focus:ring-2 focus:ring-primary-500/30
                         disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:border-bdr-medium disabled:hover:text-txt-label disabled:hover:bg-white"
                        title="Tải lên tài liệu PDF"
                        aria-label="Tải lên tài liệu PDF"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                            strokeWidth={1.8} stroke="currentColor" className="h-5 w-5">
                            <path strokeLinecap="round" strokeLinejoin="round"
                                d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5A3.375 3.375 0 0 0 10.125 2.25H7.5A2.25 2.25 0 0 0 5.25 4.5v15A2.25 2.25 0 0 0 7.5 21.75h9A2.25 2.25 0 0 0 18.75 19.5v-1.125M15.75 18h5.25m-2.625-2.625v5.25" />
                        </svg>
                    </button>

                    {/* Input area */}
                    <div className="flex-1 relative">
                        <textarea
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Nhập câu hỏi pháp lý của bạn..."
                            disabled={disabled}
                            rows={1}
                            id="chat-input"
                            className="w-full resize-none rounded-xl border border-bdr-medium bg-white
                             px-4 py-3.5 text-txt-primary text-[15px]
                             placeholder-txt-muted shadow-input
                             focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-400
                             disabled:opacity-50 disabled:cursor-not-allowed
                             max-h-48 overflow-y-auto transition-smooth"
                            style={{ minHeight: "52px" }}
                            onInput={(e) => {
                                e.target.style.height = "52px";
                                e.target.style.height =
                                    Math.min(e.target.scrollHeight, 192) + "px";
                            }}
                        />
                    </div>

                    {/* Send button */}
                    <button
                        type="submit"
                        disabled={disabled || !hasContent}
                        id="btn-send"
                        className={`flex h-11 w-11 items-center justify-center rounded-xl flex-shrink-0
                         transition-smooth focus:outline-none focus:ring-2 focus:ring-primary-400
                         ${hasContent && !disabled
                            ? "bg-primary-600 text-white shadow-md hover:bg-primary-700 hover:shadow-lg active:scale-95"
                            : "bg-gray-100 text-txt-muted border border-bdr cursor-not-allowed"
                         }`}
                        aria-label="Gửi tin nhắn"
                    >
                        {disabled ? (
                            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                        ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                                <path d="M3.478 2.404a.75.75 0 0 0-.926.941l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.404Z" />
                            </svg>
                        )}
                    </button>
                </div>

                {/* Disclaimer */}
                <p className="text-center text-[11px] text-txt-muted mt-2.5">
                    AI có thể mắc sai sót. Vui lòng kiểm tra lại thông tin quan trọng.
                </p>
            </form>
        </div>
    );
}
