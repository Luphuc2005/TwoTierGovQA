/**
 * ChatWindow — message display area with polished empty state
 * featuring suggestion cards and professional layout.
 */
import { useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import LoadingDots from "./LoadingDots";

export default function ChatWindow({ messages, isLoading }) {
    const bottomRef = useRef(null);

    // Auto-scroll to bottom when messages change or loading starts
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    // Suggestion cards with SVG icons (no emoji)
    const suggestions = [
        {
            color: "text-primary-600 bg-primary-50",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path d="M11.584 2.376a.75.75 0 0 1 .832 0l9 6a.75.75 0 1 1-.832 1.248L12 3.901 3.416 9.624a.75.75 0 0 1-.832-1.248l9-6Z" />
                    <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 0 1 0 1.5H3a.75.75 0 0 1 0-1.5h.75v-9.918a.75.75 0 0 1 .634-.74A49.109 49.109 0 0 1 12 9c2.59 0 5.134.202 7.616.592a.75.75 0 0 1 .634.74Zm-7.5 2.418a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Zm3-.75a.75.75 0 0 1 .75.75v6.75a.75.75 0 0 1-1.5 0v-6.75a.75.75 0 0 1 .75-.75ZM9 12.75a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Z" clipRule="evenodd" />
                </svg>
            ),
            text: "Chính quyền địa phương hai cấp hoạt động như thế nào?",
        },
        {
            color: "text-secondary-500 bg-teal-50",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25ZM6.262 6.072a8.25 8.25 0 1 0 10.562-.766 4.5 4.5 0 0 1-1.318 1.357L14.25 7.5l.165.33a.809.809 0 0 1-1.086 1.085l-.604-.302a1.125 1.125 0 0 0-1.298.21l-.132.131c-.439.44-.439 1.152 0 1.591l.296.296c.256.257.622.374.98.314l1.17-.195c.323-.054.654.036.905.245l1.33 1.108c.32.267.46.694.358 1.1a8.7 8.7 0 0 1-2.288 4.04l-.723.724a1.125 1.125 0 0 1-1.298.21l-.153-.076a1.125 1.125 0 0 1-.622-1.006v-1.089c0-.298-.119-.585-.33-.796l-1.347-1.347a1.125 1.125 0 0 1-.21-1.298L9.75 12l-1.64-1.64a6 6 0 0 1-1.676-3.257l-.172-1.03Z" clipRule="evenodd" />
                </svg>
            ),
            text: "Luật Tổ chức chính quyền địa phương quy định gì?",
        },
        {
            color: "text-amber-600 bg-amber-50",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd" d="M5.625 1.5c-1.036 0-1.875.84-1.875 1.875v17.25c0 1.035.84 1.875 1.875 1.875h12.75c1.035 0 1.875-.84 1.875-1.875V12.75A3.75 3.75 0 0 0 16.5 9h-1.875a1.875 1.875 0 0 1-1.875-1.875V5.25A3.75 3.75 0 0 0 9 1.5H5.625ZM7.5 15a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5A.75.75 0 0 1 7.5 15Zm.75 2.25a.75.75 0 0 0 0 1.5H12a.75.75 0 0 0 0-1.5H8.25Z" clipRule="evenodd" />
                    <path d="M12.971 1.816A5.23 5.23 0 0 1 14.25 5.25v1.875c0 .207.168.375.375.375H16.5a5.23 5.23 0 0 1 3.434 1.279 9.768 9.768 0 0 0-6.963-6.963Z" />
                </svg>
            ),
            text: "Nhiệm vụ và quyền hạn của UBND cấp huyện?",
        },
        {
            color: "text-violet-600 bg-violet-50",
            icon: (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd" d="M4.125 3C3.089 3 2.25 3.84 2.25 4.875V18a3 3 0 0 0 3 3h15a3 3 0 0 1-3-3V4.875C17.25 3.839 16.41 3 15.375 3H4.125ZM12 9.75a.75.75 0 0 0 0 1.5h1.5a.75.75 0 0 0 0-1.5H12Zm-.75-2.25a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5H12a.75.75 0 0 1-.75-.75ZM6 12.75a.75.75 0 0 0 0 1.5h7.5a.75.75 0 0 0 0-1.5H6Zm-.75 3.75a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5H6a.75.75 0 0 1-.75-.75ZM6 6.75a.75.75 0 0 0-.75.75v3c0 .414.336.75.75.75h3a.75.75 0 0 0 .75-.75v-3A.75.75 0 0 0 9 6.75H6Z" clipRule="evenodd" />
                    <path d="M18.75 6.75h1.875c.621 0 1.125.504 1.125 1.125V18a1.5 1.5 0 0 1-3 0V6.75Z" />
                </svg>
            ),
            text: "Nghị quyết 35/2023/UBTVQH15 có nội dung gì?",
        },
    ];

    return (
        <div className="flex-1 overflow-y-auto bg-surface bg-pattern">
            {messages.length === 0 && !isLoading ? (
                /* ── Empty State ───────────────────────────── */
                <div className="flex flex-col items-center justify-center h-full px-6 animate-fade-in">
                    <div className="max-w-xl w-full text-center">
                        {/* Icon */}
                        <div className="w-[72px] h-[72px] rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700
                             flex items-center justify-center mx-auto mb-5 shadow-lg">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"
                                className="w-9 h-9 text-white">
                                <path d="M11.584 2.376a.75.75 0 0 1 .832 0l9 6a.75.75 0 1 1-.832 1.248L12 3.901 3.416 9.624a.75.75 0 0 1-.832-1.248l9-6Z" />
                                <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 0 1 0 1.5H3a.75.75 0 0 1 0-1.5h.75v-9.918a.75.75 0 0 1 .634-.74A49.109 49.109 0 0 1 12 9c2.59 0 5.134.202 7.616.592a.75.75 0 0 1 .634.74Zm-7.5 2.418a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Zm3-.75a.75.75 0 0 1 .75.75v6.75a.75.75 0 0 1-1.5 0v-6.75a.75.75 0 0 1 .75-.75ZM9 12.75a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Z" clipRule="evenodd" />
                            </svg>
                        </div>

                        {/* Welcome text */}
                        <h2 className="text-[26px] font-extrabold text-txt-primary mb-2 tracking-tight">
                            Trợ Lý Pháp Lý AI
                        </h2>
                        <p className="text-txt-secondary text-[15px] mb-8 leading-relaxed max-w-sm mx-auto">
                            Hỗ trợ tra cứu và giải đáp văn bản pháp luật Việt Nam
                            về chính quyền địa phương hai cấp
                        </p>

                        {/* Suggestion cards */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
                            {suggestions.map((item, idx) => (
                                <button
                                    key={idx}
                                    className="flex items-start gap-3.5 p-4 rounded-xl bg-white border border-bdr
                                     hover:border-primary-300 hover:shadow-card-lg hover:-translate-y-0.5
                                     transition-smooth text-left group shadow-card"
                                >
                                    <div className={`w-9 h-9 rounded-lg ${item.color} flex items-center justify-center flex-shrink-0`}>
                                        {item.icon}
                                    </div>
                                    <span className="text-sm text-txt-secondary group-hover:text-txt-primary transition-smooth leading-relaxed pt-1.5">
                                        {item.text}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            ) : (
                /* ── Messages ─────────────────────────────── */
                <div className="max-w-3xl mx-auto py-6 px-4 md:px-6">
                    {messages.map((msg) => (
                        <MessageBubble
                            key={msg.id}
                            role={msg.role}
                            content={msg.content}
                            timestamp={msg.created_at}
                        />
                    ))}
                    {isLoading && <LoadingDots />}
                    <div ref={bottomRef} />
                </div>
            )}
        </div>
    );
}
