/**
 * ProtectedRoute — redirects to /login if user is not authenticated.
 */
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ProtectedRoute({ children }) {
    const { user, loading } = useAuth();

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-screen bg-surface">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary-600 to-primary-700
                     flex items-center justify-center mb-4 shadow-lg animate-pulse">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-white">
                        <path d="M11.584 2.376a.75.75 0 0 1 .832 0l9 6a.75.75 0 1 1-.832 1.248L12 3.901 3.416 9.624a.75.75 0 0 1-.832-1.248l9-6Z" />
                        <path fillRule="evenodd" d="M20.25 10.332v9.918H21a.75.75 0 0 1 0 1.5H3a.75.75 0 0 1 0-1.5h.75v-9.918a.75.75 0 0 1 .634-.74A49.109 49.109 0 0 1 12 9c2.59 0 5.134.202 7.616.592a.75.75 0 0 1 .634.74Zm-7.5 2.418a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Zm3-.75a.75.75 0 0 1 .75.75v6.75a.75.75 0 0 1-1.5 0v-6.75a.75.75 0 0 1 .75-.75ZM9 12.75a.75.75 0 0 0-1.5 0v6.75a.75.75 0 0 0 1.5 0v-6.75Z" clipRule="evenodd" />
                    </svg>
                </div>
                <div className="loading-dots flex space-x-2">
                    <span className="w-2.5 h-2.5 bg-primary-400 rounded-full inline-block" />
                    <span className="w-2.5 h-2.5 bg-primary-400 rounded-full inline-block" />
                    <span className="w-2.5 h-2.5 bg-primary-400 rounded-full inline-block" />
                </div>
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return children;
}
