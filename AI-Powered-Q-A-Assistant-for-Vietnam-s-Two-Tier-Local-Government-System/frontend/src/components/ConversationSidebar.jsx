/**
 * ConversationSidebar — light-themed sidebar with high contrast text,
 * search bar, conversation history, and user profile section.
 */
import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function ConversationSidebar({
    conversations,
    activeId,
    onSelect,
    onNew,
    onDelete,
}) {
    const { user, logout } = useAuth();
    const [searchQuery, setSearchQuery] = useState("");

    const filtered = conversations.filter((conv) =>
        conv.title.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <aside className="w-[280px] bg-white flex flex-col h-screen border-r border-bdr">
            {/* ── Brand Header ─────────────────────────────── */}
            <div className="px-5 pt-5 pb-4 border-b border-bdr">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-primary-600 flex items-center justify-center shadow-md flex-shrink-0">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-white">
                            <path d="M11.584 2.376a.75.75 0 0 1 .832 0l9 6a.75.75 0 1 1-.832 1.248L12 3.901 3.416 9.624a.75.75 0 0 1-.832-1.248l9-6Z" />
                            <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 0 1 0 1.5H3a.75.75 0 0 1 0-1.5h.75v-9.918a.75.75 0 0 1 .634-.74A49.109 49.109 0 0 1 12 9c2.59 0 5.134.202 7.616.592a.75.75 0 0 1 .634.74Zm-7.5 2.418a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Zm3-.75a.75.75 0 0 1 .75.75v6.75a.75.75 0 0 1-1.5 0v-6.75a.75.75 0 0 1 .75-.75ZM9 12.75a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Z" clipRule="evenodd" />
                        </svg>
                    </div>
                    <div>
                        <h1 className="text-[15px] font-bold text-txt-primary leading-tight">
                            Legal Chatbot
                        </h1>
                        <p className="text-xs text-txt-label leading-tight">
                            Trợ lý pháp lý AI
                        </p>
                    </div>
                </div>
            </div>

            {/* ── New Chat Button ──────────────────────────── */}
            <div className="px-4 pt-3 pb-3">
                <button
                    onClick={onNew}
                    id="btn-new-chat"
                    className="w-full flex items-center justify-center gap-2 rounded-xl
                     bg-primary-600 hover:bg-primary-700
                     px-4 py-2.5 text-[13px] font-semibold text-white
                     transition-smooth shadow-md hover:shadow-lg
                     active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-primary-400 focus:ring-offset-2"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                        <path d="M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z" />
                    </svg>
                    Cuộc trò chuyện mới
                </button>
            </div>

            {/* ── Search Bar ──────────────────────────────── */}
            <div className="px-4 pb-3">
                <div className="relative">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                        className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-txt-muted"
                    >
                        <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z" clipRule="evenodd" />
                    </svg>
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Tìm kiếm hội thoại..."
                        className="w-full rounded-lg bg-surface border border-bdr
                         pl-9 pr-3 py-2.5 text-sm text-txt-primary placeholder-txt-muted
                         focus:outline-none focus:ring-2 focus:ring-primary-500/30 focus:border-primary-400
                         transition-smooth"
                    />
                </div>
            </div>

            {/* ── Conversation List ───────────────────────── */}
            <div className="flex-1 overflow-y-auto px-3">
                <p className="text-[11px] font-semibold text-txt-label uppercase tracking-wider px-2 mb-2">
                    Lịch sử trò chuyện
                </p>

                <div className="space-y-0.5">
                    {filtered.map((conv) => (
                        <div
                            key={conv.id}
                            id={`conv-${conv.id}`}
                            className={`group flex items-center rounded-xl px-3 py-2.5 cursor-pointer
                            transition-smooth ${
                                activeId === conv.id
                                    ? "bg-sidebar-active text-primary-700 font-semibold shadow-sm"
                                    : "text-txt-primary hover:bg-sidebar-hover"
                            }`}
                            onClick={() => onSelect(conv.id)}
                        >
                            {/* Chat icon */}
                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mr-2.5
                                ${activeId === conv.id
                                    ? 'bg-primary-100'
                                    : 'bg-gray-100'
                                }`}
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"
                                    className={`w-4 h-4 ${activeId === conv.id ? 'text-primary-600' : 'text-txt-label'}`}>
                                    <path fillRule="evenodd" d="M3.43 2.524A41.29 41.29 0 0 1 10 2c2.236 0 4.43.16 6.57.524 1.437.245 2.43 1.564 2.43 3.015v2.923c0 1.45-.993 2.77-2.43 3.015a41.307 41.307 0 0 1-2.825.39.75.75 0 0 0-.543.317l-1.827 2.669a.75.75 0 0 1-1.238 0l-1.828-2.67a.75.75 0 0 0-.543-.316 41.29 41.29 0 0 1-2.825-.39C1.993 11.232 1 9.912 1 8.463V5.538c0-1.45.993-2.77 2.43-3.015Z" clipRule="evenodd" />
                                </svg>
                            </div>

                            <span className="truncate flex-1 text-[13px]">{conv.title}</span>

                            {/* Delete button */}
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDelete(conv.id);
                                }}
                                className="opacity-0 group-hover:opacity-100 ml-1 p-1.5 rounded-lg
                                 hover:bg-red-50 transition-smooth"
                                title="Xóa cuộc trò chuyện"
                                aria-label="Xóa cuộc trò chuyện"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"
                                    className="w-3.5 h-3.5 text-txt-muted hover:text-red-500">
                                    <path fillRule="evenodd" d="M8.75 1A2.75 2.75 0 0 0 6 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 1 0 .23 1.482l.149-.022.841 10.518A2.75 2.75 0 0 0 7.596 19h4.807a2.75 2.75 0 0 0 2.742-2.53l.841-10.519.149.023a.75.75 0 0 0 .23-1.482A41.03 41.03 0 0 0 14 4.193V3.75A2.75 2.75 0 0 0 11.25 1h-2.5ZM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4ZM8.58 7.72a.75.75 0 0 0-1.5.06l.3 7.5a.75.75 0 1 0 1.5-.06l-.3-7.5Zm4.34.06a.75.75 0 1 0-1.5-.06l-.3 7.5a.75.75 0 1 0 1.5.06l.3-7.5Z" clipRule="evenodd" />
                                </svg>
                            </button>
                        </div>
                    ))}
                </div>

                {filtered.length === 0 && conversations.length > 0 && (
                    <div className="text-center py-8">
                        <p className="text-txt-muted text-sm">
                            Không tìm thấy kết quả
                        </p>
                    </div>
                )}

                {conversations.length === 0 && (
                    <div className="text-center py-10 px-4">
                        <div className="w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center mx-auto mb-3">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 text-txt-muted">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 0 1-.825-.242m9.345-8.334a2.126 2.126 0 0 0-.476-.095 48.64 48.64 0 0 0-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0 0 11.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155" />
                            </svg>
                        </div>
                        <p className="text-txt-label text-sm font-medium mb-1">
                            Chưa có cuộc trò chuyện
                        </p>
                        <p className="text-txt-muted text-xs">
                            Nhấn nút ở trên để bắt đầu
                        </p>
                    </div>
                )}
            </div>

            {/* ── User Footer ─────────────────────────────── */}
            <div className="border-t border-bdr p-3">
                <div className="flex items-center justify-between rounded-xl px-3 py-2.5">
                    <div className="flex items-center gap-3 min-w-0">
                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center flex-shrink-0 shadow-sm">
                            <span className="text-sm font-bold text-white">
                                {user?.email?.charAt(0).toUpperCase() || "U"}
                            </span>
                        </div>
                        <div className="min-w-0">
                            <p className="text-sm font-semibold text-txt-primary truncate leading-tight">
                                {user?.full_name || "Người dùng"}
                            </p>
                            <p className="text-xs text-txt-label truncate leading-tight">
                                {user?.email || ""}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Logout button — full width, clearly visible */}
                <button
                    onClick={logout}
                    id="btn-logout"
                    className="w-full mt-2 flex items-center justify-center gap-2 rounded-xl
                     px-3 py-2.5 text-[13px] font-medium text-red-600
                     bg-red-50 border border-red-200
                     hover:bg-red-100 hover:text-red-700 hover:border-red-300
                     transition-smooth
                     focus:outline-none focus:ring-2 focus:ring-red-400/30"
                    title="Đăng xuất"
                    aria-label="Đăng xuất"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                        <path fillRule="evenodd" d="M3 4.25A2.25 2.25 0 0 1 5.25 2h5.5A2.25 2.25 0 0 1 13 4.25v2a.75.75 0 0 1-1.5 0v-2a.75.75 0 0 0-.75-.75h-5.5a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h5.5a.75.75 0 0 0 .75-.75v-2a.75.75 0 0 1 1.5 0v2A2.25 2.25 0 0 1 10.75 18h-5.5A2.25 2.25 0 0 1 3 15.75V4.25Z" clipRule="evenodd" />
                        <path fillRule="evenodd" d="M19 10a.75.75 0 0 0-.75-.75H8.704l1.048-.943a.75.75 0 1 0-1.004-1.114l-2.5 2.25a.75.75 0 0 0 0 1.114l2.5 2.25a.75.75 0 1 0 1.004-1.114l-1.048-.943h9.546A.75.75 0 0 0 19 10Z" clipRule="evenodd" />
                    </svg>
                    Đăng xuất
                </button>
            </div>
        </aside>
    );
}
