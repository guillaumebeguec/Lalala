import { Logo } from "./Logo";

export function Header({ action, onLogoLongPress }: { action?: { label: string; onClick: () => void }; onLogoLongPress?: () => void }) {
  return (
    <header className="flex items-center justify-between py-2">
      <Logo onLongPress={onLogoLongPress} />
      {action && <button type="button" onClick={action.onClick} className="min-h-[44px] rounded-pill px-4 text-[15px] font-medium text-text-2 hover:text-text">{action.label}</button>}
    </header>
  );
}
