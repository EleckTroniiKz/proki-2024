 # Roadmap JUUUNGE

 * Globals
    * Clean Code (sonst fetzen)
    * Fokus Computer Vision wenig fokus auf AI
    * Wöchentliche Meetings (Fix Meeting Montag, falls nicht, dann wird ein anderer Tag besprochen)

 * Pre-Processing
    * Bilder einlesen (Part, Gripper) [Can]
    * Kurz-Evaluierung geeigneter Libraries
    * Parts Masks erstellen (Edge Detection, ...) [Lukas]

 * Algorithmus [Torben] [Philip]
    * TBD -> Gripper positioning and rotation
        * Definition von Algorithmus (alle maybe?)
        * Output: gripper, part, x, y, winkel
      
    * Optional (Visualisierung des Ablaufs) -> wird langsamer aber wir sehen was abgeht
    * Time Restrictions beachten (pro Part Gripper nicht mehr als 3 Sekunden)

 * Post-Processing
    * Ausgabe Datei erstellen
    * Code dokumentieren+

* Algorithmus Idee:
    * Schwarz Weiß Convertierung

    * (Schwerpunkt bestimmen
        * Gesamt shape bestimmen
        * Lücken subtrahieren, um Schwerpunkt zu verschieben
          (durch Library im Idealfall machen lassen, ansonsten überlegen durch integrale)
        * dabei Punkt (0,0) beachten
        * Visualisierung des COM (center of mass)) doch nicht, einfach durch center of image approximieren, warum auch immer
    
    * Gripper möglichst in Mitte des Images setzen (Bewertungskriterium)
    * 2D Bit array ertstellen und testen, ob gripper Position gültig

      * Position bestimmen
        * gripper in Schwerpunkt setzen und um 360° rotieren, bis möglich
        * falls nicht auffindbar, anderer Punkt als Schwerpunkt nutzen

* Part, Gripper ändern



