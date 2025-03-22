import { exampleThemeStorage } from '@extension/storage';
import { useStorage } from '@extension/shared';
import type { ComponentPropsWithoutRef } from 'react';
import { cn } from '@/lib/utils';

type ToggleButtonProps = ComponentPropsWithoutRef<'button'>;

export const ToggleButton = ({ className, children, ...props }: ToggleButtonProps) => {
  const theme = useStorage(exampleThemeStorage);

  return (
    <button
      className={cn(
        className,
        'rounded shadow hover:scale-105 text-2xl',
        theme === 'light' ? 'border-black' : 'border-white',
        'border-2 font-bold',
      )}
      onClick={exampleThemeStorage.toggle}>
      {theme === 'dark' ? '🌙' : '☀️'}
    </button>
  );
};
