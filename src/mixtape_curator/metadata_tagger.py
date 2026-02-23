
import cmd
from .library import library

class MetadataTagger(cmd.Cmd):
    intro = 'Welcome to the Metadata Tagger. Type "help" to see commands.'
    prompt = '(tagger) '

    def do_fill_years(self, arg):
        """Iterate tracks with missing release year."""
        if 'release_year' not in library.df.columns:
            library.df['release_year'] = None
            
        missing = library.df[library.df['release_year'].isnull() | library.df['release_year'].isna()]
        total = len(missing)
        print(f"Found {total} tracks missing Release Year.")
        
        count = 0
        for idx, row in missing.iterrows():
            count += 1
            print(f"\n[{count}/{total}] {row['title']} - {row['artist']}")
            val = input("Year (Enter to skip, 'q' to quit): ").strip()
            
            if val.lower() == 'q':
                break
            
            if val.isdigit():
                library.df.at[idx, 'release_year'] = int(val)
                print("Saved.")
            else:
                print("Skipped.")
                
        self.do_save("")

    def do_rate_vibe(self, arg):
        """Iterate all tracks to rate Energy/Valence."""
        # TODO: Implement optional filtering
        print("Starting Vibe Rating (Energy 0.0-1.0, Valence 0.0-1.0)")
        
        for idx, row in library.df.iterrows():
            print(f"\n{row['title']} - {row['artist']}")
            print(f"Current: Energy={row.get('energy', 0.0):.2f}, Valence={row.get('valence', 0.0):.2f}")
            
            e = input("New Energy (Enter to skip, 'q' to quit): ").strip()
            if e.lower() == 'q': break
            if e:
                try:
                    library.df.at[idx, 'energy'] = float(e)
                except ValueError: pass
                
            v = input("New Valence (Enter to skip): ").strip()
            if v:
                try:
                    library.df.at[idx, 'valence'] = float(v)
                except ValueError: pass
                
        self.do_save("")

    def do_save(self, arg):
        """Save changes to library.json."""
        library.save()

    def do_quit(self, arg):
        """Exit."""
        return True
        
    def do_EOF(self, arg):
        return True

if __name__ == '__main__':
    library.load()
    MetadataTagger().cmdloop()
