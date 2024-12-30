 # Roadmap JUUUNGE

 * Globals
    * Clean Code (sonst fetzen)
    * Fokus Computer Vision wenig fokus auf AI
    * Wöchentliche Meetings (Fix Meeting Montag, falls nicht, dann wird ein anderer Tag besprochen)

 * Pre-Processing
    * Bilder einlesen (Part, Gripper) [Done; -> Optimierung und testing von Can ToDO]
    * Kurz-Evaluierung geeigneter Libraries
    * Parts Masks erstellen (Edge Detection, ...) [Lukas und Can]

 * Algorithmus [Torben] [Philip] -> ausgewählter Ansatz: Basically optimierte Brute Forcing
    * TBD -> Gripper positioning and rotation
        * Definition von Algorithmus (alle maybe?)
        * Output: gripper, part, x, y, winkel
      
    * Optional (Visualisierung des Ablaufs) -> wird langsamer aber wir sehen was abgeht
    * Time Restrictions beachten (pro Part Gripper nicht mehr als 3 Sekunden)

 * Post-Processing
    * Ausgabe Datei erstellen
    * Code dokumentieren+




* load binary mask aus evaluate

* Rastersuche:
   * Beginnen in der Mitte des Parts, dann nach oben / links / rechts / unten, und dann 2. Ebene, bis Lösung gefunden wurde, falls diese        gefunden wird, gilt dies als unsere optimale Lösung; Stop Regel für Gripper Mittelpunkt (falls dieser näher an Rand ist wie dieser breit / lang, dann abbrechen bzw andere Line überprüfen)

   * Am Anfang bei Gripper auf Part: Anzahl Mittelpunkte der Gripper Saugnäpfe herausfinden (1, 2 oder 4); 2. Array erstellen, dort für einen Gripper Saugknopf einmal durchiterieren und markieren, ob dieser eine Punkt (oder alle 2 / 4) eine mögliche Mitte wäre.
   Bereiche am Rand des Parts können direkt zu Beginn als false markiert werden
   --> es kann in O(1) überprüft werden, ob ein Grippersaugknopf bei diesem Mittelpunkt valide ist --> Überprüfung O(Anzahl Saugknöpfe)
   Problem: müsste man für alle Winkel machen, also man bräuchte 360 solcher Arrays

   * prüfen wie viel Grad man prüfen muss, maximal immer nur 180 Grad das die Gripper symmetrisch sind.

   * Bildqualität verringern: 4 Pixel -> 1 Pixel, und dann algorithmus auf diesem datenset laufen lassen, bei passendem Ergebnis genauer auf originalem Datensatz abarbeiten


   * calc new angle index (depending on how many pixels overlap)


### Fragen
* Sollen wir eigene Gripper erstellen oder sollen wir die vorhandenen benutzen?
  * Auf jeden Fall die vorhanden Gripper benutzten und auch mit denen testen, es sollte aber auch generisch für beliebige  nue Gripperarten funktionieren

* Wo ist der Startpunkt des blauen Punktes
  * baluer Punkte ist 0 Grad auf der Waagerechte
  * Grad wächst im Uhrzeigersinn -> siehe readMe
* 
* Warum sind ganz unten keine Gripper gegeben?
  * keine Gripper vorgeschrieben, einfach Daten mit denen wir noch weiter ausprobieren können
* 
* Gripper selber erstellen oder so?
  * kann man machen um auf generische funktionalität zu testen

* Evaluate Ordner -> wird noch verbessert! 

* position is_valid methode

* Visualization:
    * Plot für Gripper auf Part (unter Beachtung des Winkels)




