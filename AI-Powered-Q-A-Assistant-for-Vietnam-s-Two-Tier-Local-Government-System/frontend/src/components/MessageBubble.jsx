/**
 * MessageBubble — chat message with card-style AI responses,
 * blue user bubbles, clear avatars, and timestamps.
 */
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function MessageBubble({ role, content, timestamp }) {
    const isUser = role === "user";

    const formatTime = (ts) => {
        if (!ts) return "";
        try {
            let dateStr = String(ts);
            // Backend returns UTC timestamps without 'Z' suffix — fix it
            if (dateStr.includes("T") && !dateStr.endsWith("Z") && !/[+-]\d{2}:\d{2}$/.test(dateStr)) {
                dateStr += "Z";
            }
            const date = new Date(dateStr);
            return date.toLocaleTimeString("vi-VN", {
                hour: "2-digit",
                minute: "2-digit",
            });
        } catch {
            return "";
        }
    };

    return (
        <div
            className={`flex gap-3 mb-5 animate-fade-in ${
                isUser ? "flex-row-reverse" : "flex-row"
            }`}
        >
            {/* Avatar */}
            <div
                className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm ${
                    isUser
                        ? "bg-primary-600"
                        : "bg-gradient-to-br from-emerald-500 to-teal-600"
                }`}
            >
                <span className="text-xs font-bold text-white">
                    {isUser ? "U" : "AI"}
                </span>
            </div>

            {/* Message Content */}
            <div className={`max-w-[78%] flex flex-col ${isUser ? "items-end" : "items-start"}`}>
                <div
                    className={`rounded-2xl px-5 py-3.5 ${
                        isUser
                            ? "bg-primary-600 text-white rounded-tr-sm"
                            : "bg-white border border-bdr shadow-card rounded-tl-sm"
                    }`}
                >
                    {isUser ? (
                        <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
                            {content}
                        </p>
                    ) : (
                        <div className="markdown-body">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {content}
                            </ReactMarkdown>
                        </div>
                    )}
                </div>

                {/* Timestamp */}
                {timestamp && (
                    <p className="text-[11px] text-txt-muted mt-1.5 px-1">
                        {formatTime(timestamp)}
                    </p>
                )}
            </div>
        </div>
    );
}
