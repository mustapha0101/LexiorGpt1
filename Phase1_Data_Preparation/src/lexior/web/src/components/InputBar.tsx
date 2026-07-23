import { useRef, useState } from "react";

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
  streaming: boolean;
  onCancel: () => void;
  placeholder?: string;
}

export function InputBar({
  onSend,
  disabled,
  streaming,
  onCancel,
  placeholder = "Ask a legal question...",
}: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSubmit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleInput() {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  }

  return (
    <div className="border-t border-border bg-surface px-4 py-3">
      <div className="max-w-3xl mx-auto flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            placeholder={placeholder}
            rows={1}
            disabled={disabled}
            className="
              w-full resize-none rounded-xl border border-border bg-surface-alt
              px-4 py-3 text-sm text-text-primary placeholder-text-muted
              focus:outline-none focus:ring-2 focus:ring-brand-500/40 focus:border-brand-500
              disabled:opacity-50 transition-shadow
            "
          />
        </div>

        {streaming ? (
          <button
            onClick={onCancel}
            className="
              shrink-0 h-11 px-4 rounded-xl text-sm font-medium
              bg-error/10 text-error hover:bg-error/20
              transition-colors cursor-pointer
            "
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!value.trim() || disabled}
            className="
              shrink-0 h-11 px-4 rounded-xl text-sm font-medium text-white
              bg-brand-600 hover:bg-brand-700 disabled:opacity-40
              transition-colors cursor-pointer disabled:cursor-not-allowed
            "
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}
