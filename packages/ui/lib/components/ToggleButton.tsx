import { exampleThemeStorage } from '@extension/storage';
import { useStorage } from '@extension/shared';
import type { ComponentPropsWithoutRef } from 'react';

type ToggleButtonProps = ComponentPropsWithoutRef<'button'>;

export const ToggleButton = ({ className, children, ...props }: ToggleButtonProps) => {
  const theme = useStorage(exampleThemeStorage);

  return (
    <button className={`w-full text-left text-sm`} onClick={exampleThemeStorage.toggle}>
      {theme === 'light' ? 'Dark Mode 🌙' : 'Light Mode ☀️'}
    </button>
  );
};
