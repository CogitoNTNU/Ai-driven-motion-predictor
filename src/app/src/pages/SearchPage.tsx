import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";

interface SearchResult {
  symbol: string;
  name: string;
  type: string;
}

export function SearchPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const [showDropdown, setShowDropdown] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchResults = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      setShowDropdown(false);
      return;
    }
    setIsLoading(true);
    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL ?? ""}/api/search?q=${encodeURIComponent(q)}`
      );
      if (res.ok) {
        const data: SearchResult[] = await res.json();
        setResults(data.slice(0, 8));
        setShowDropdown(data.length > 0);
        setHighlightedIndex(-1);
      }
    } catch (err) {
      console.error("Search error:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => {
      fetchResults(query);
    }, 300);
    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, [query, fetchResults]);

  const navigateToStock = (symbol: string) => {
    navigate(`/stock/${symbol}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown) {
      if (e.key === "Enter" && query.trim()) {
        navigateToStock(query.trim().toUpperCase());
      }
      return;
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.min(prev + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((prev) => Math.max(prev - 1, -1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (highlightedIndex >= 0 && results[highlightedIndex]) {
        navigateToStock(results[highlightedIndex].symbol);
      } else if (query.trim()) {
        navigateToStock(query.trim().toUpperCase());
      }
    } else if (e.key === "Escape") {
      setShowDropdown(false);
      setHighlightedIndex(-1);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#212121] px-4">
      <div className="w-full max-w-xl">
        {/* Title */}
        <h1 className="mb-2 text-center text-4xl font-bold text-white">
          AI-Driven Stock Predictor
        </h1>
        <p className="mb-10 text-center text-base text-[#9ca3af]">
          Enter a ticker to get started
        </p>

        {/* Search container */}
        <div className="relative">
          <div className="relative flex items-center">
            <Search className="absolute left-4 h-5 w-5 text-[#9ca3af]" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => results.length > 0 && setShowDropdown(true)}
              onBlur={() => setTimeout(() => setShowDropdown(false), 150)}
              placeholder="Search ticker or company name..."
              autoFocus
              className="w-full rounded-full border border-[#4d4d4f] bg-[#2f2f2f] py-4 pl-12 pr-5 text-base text-white placeholder:text-[#6b7280] outline-none transition-all focus:border-[#10a37f] focus:ring-2 focus:ring-[#10a37f]/30"
            />
            {isLoading && (
              <div className="absolute right-4 h-4 w-4 animate-spin rounded-full border-2 border-[#10a37f] border-t-transparent" />
            )}
          </div>

          {/* Dropdown results */}
          {showDropdown && results.length > 0 && (
            <div className="absolute top-full z-50 mt-2 w-full overflow-hidden rounded-xl border border-[#4d4d4f] bg-[#2f2f2f] shadow-2xl">
              {results.map((result, index) => (
                <button
                  key={result.symbol}
                  onMouseDown={() => navigateToStock(result.symbol)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                  className={`flex w-full items-center gap-3 px-4 py-3 text-left transition-colors ${
                    index === highlightedIndex
                      ? "bg-[#10a37f]/10"
                      : "hover:bg-[#404040]"
                  } ${index < results.length - 1 ? "border-b border-[#4d4d4f]/50" : ""}`}
                >
                  <div className="flex min-w-0 flex-1 items-center gap-3">
                    <span className="shrink-0 font-bold text-white">
                      {result.symbol}
                    </span>
                    <span className="truncate text-sm text-[#9ca3af]">
                      {result.name}
                    </span>
                  </div>
                  <span className="shrink-0 rounded px-1.5 py-0.5 text-xs text-[#9ca3af] border border-[#4d4d4f]">
                    {result.type}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
