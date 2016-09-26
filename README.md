# Lean 'n' Recognize
Détecte, analyse et compare des visages sur des images issues de différentes sources

Cette image est le résultat fonctionnel du tout premier algorithme de ma fabrication utilisant le mécanisme de Haar Cascade, basé sur les travaux de [Natoshi Seo](http://note.sonots.com/SciSoftware/haartraining.html):

![alt tag ](https://github.com/blackrainb0w/Lean-n-Recognize/raw/master/testCam/proof.png "Learn 'n' Recognize V0")
<p style="text-align: center; font-style: italic;">Learn 'n' Recognize V0 (testCam) "proof of concept"</p>

Cette image est le résultat fonctionnel de la version 3 de Learn 'n' Recognize. Cette version reprend certains principes des version 0, 1 et 2 tout en utilisant comme image d'entré un flux vidéo issue d'une webcam. Les algorithme à partir de la version 1 utilisent [LBPH (Local Binary Pattern Histogram)](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms). Il s'agit d'un algorithme permettant de transcrire des zones d'une image en nombres décimaux afin d'analyser des [patternes](https://www.wikiwand.com/fr/Pattern), comme des formes, permettant ainsi de discriminer les visages.

![alt tag ](https://github.com/blackrainb0w/Lean-n-Recognize/raw/master/learn_n_recognize_V3/proof.png "Learn 'n' Recognize V3")
<p style="text-align: center; font-style: italic;">Learn 'n' Recognize V3 "proof of concept"</p>

La version 5 (actuellement la dernière en date) allie la version 3, permettant de détecter et analyser en directe un visage, et la version 4, permettant l'apprentissage d'un visage (en l'associant à un nom), en proposant ces fonctionnalités sous formes de deux modes. Il est alors possible, d'apprendre un visage en l'associant à un nom (mode apprentissage) et ensuite de le distinguer parmis plusieurs autres visage (mode scan); le tout en une seule et unique application.

Voici un "proof of concept" en vidéo de la version 5 (cliquez sur la vignette pour ouvrir YouTube.com):

[![Learn 'n' Recognize V5 - Proof of concept](http://img.youtube.com/vi/9-2KMnhYZOk/0.jpg)](https://www.youtube.com/watch?v=9-2KMnhYZOk "Learn 'n' Recognize V5 - Proof of concept ")

Learn 'n' Recognize est distribué sous [licence Apache Version 2.0](http://www.apache.org/licenses/)