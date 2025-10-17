# streamlit_match_analysis_begles.py
# Code adapté pour Streamlit 1.50.0 (version ancienne)
# Ajout de la fonctionnalité d'export de playlist clips

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import numpy as np 
import re 

# --- Configuration de la page ---
st.set_page_config(page_title="Analyse match - Carquefou HB", layout="wide")

st.title("🤾‍♀️ Tableau de bord - Analyse des actions Handball")

# ---------------------- Chargement du CSV ----------------------
uploaded_file = st.file_uploader("📂 Déposez votre fichier CSV", type=["csv"])

# Initialisation du DataFrame à None et des variables globales
df = None 
taux_reussite = 0
taux_parade = 0
fig_compare = None
fig_heatmap = None
all_joueurs = []
youtube_url_global = "" # Variable pour stocker l'URL globale

def read_flex_csv(file):
    """Tente de lire le CSV avec différents séparateurs et encodages."""
    file.seek(0)
    for sep in [",", ";", "\t"]:
        file.seek(0) # Réinitialiser pour chaque tentative
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df_temp = pd.read_csv(file, sep=sep, encoding=encoding)
                if df_temp.shape[1] > 1:
                    file.seek(0) 
                    return df_temp
            except Exception:
                continue
    file.seek(0)
    return None

def convert_time_to_seconds(time_str):
    """Convertit un timecode MM:SS ou SS en secondes."""
    if pd.isna(time_str) or time_str == '':
        return 0
    try:
        return float(time_str)
    except ValueError:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            try:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return int(minutes * 60 + seconds)
            except ValueError:
                return 0
        return 0

# --- Initialisation de l'état de session pour la vidéo et la playlist ---
if 'current_clip_time' not in st.session_state:
    st.session_state.current_clip_time = None
if 'playlist' not in st.session_state:
    st.session_state.playlist = []
if 'current_playlist_index' not in st.session_state:
    st.session_state.current_playlist_index = -1 
if 'select_all_state' not in st.session_state:
    st.session_state.select_all_state = False 

# --- Fonctions de rappel pour la playlist ---

def set_clip_time(time_in_seconds, is_playlist=False):
    """Sélectionne un clip pour la lecture."""
    st.session_state.current_clip_time = time_in_seconds
    if not is_playlist and st.session_state.current_playlist_index != 0:
        st.session_state.current_playlist_index = -1 
    st.rerun() 

def toggle_select_all(data_to_display):
    """Bascule la sélection de toutes les actions dans le tableau."""
    
    st.session_state.select_all_state = not st.session_state.select_all_state
    
    for idx, row in data_to_display.iterrows():
        checkbox_key = f"select_clip_{idx}"
        st.session_state[checkbox_key] = st.session_state.select_all_state

    st.rerun()


def create_and_start_playlist(filtered_df):
    """Crée la playlist et lance le premier clip immédiatement."""
    playlist_times = []
    
    temp_df_for_checkbox = filtered_df.reset_index(drop=True)
    
    for i, temp_row in temp_df_for_checkbox.iterrows():
        if st.session_state.get(f"select_clip_{i}", False):
            playlist_times.append(temp_row["Secondes de jeu"])


    if playlist_times:
        st.session_state.playlist = playlist_times
        st.session_state.current_playlist_index = 0
        new_time = playlist_times[0]
        st.session_state.current_clip_time = new_time
        st.success(f"Playlist créée et lancée : {len(st.session_state.playlist)} clips. Veuillez cliquer sur 'Play' (▶️) si la lecture ne démarre pas.")
    else:
        st.warning("Aucune action n'a été cochée. La Playlist est vide.")
        st.session_state.playlist = []
        st.session_state.current_playlist_index = -1
        st.session_state.current_clip_time = None
    st.rerun()


def play_next_clip():
    """Passe au clip suivant dans la playlist."""
    if st.session_state.playlist and st.session_state.current_playlist_index >= 0:
        if st.session_state.current_playlist_index + 1 < len(st.session_state.playlist):
            st.session_state.current_playlist_index += 1
            new_time = st.session_state.playlist[st.session_state.current_playlist_index]
            set_clip_time(new_time, is_playlist=True)
        else:
            st.session_state.current_clip_time = None
            st.session_state.current_playlist_index = len(st.session_state.playlist)
            st.success("Fin de la playlist. Toutes les actions ont été jouées.")

def play_prev_clip():
    """Passe au clip précédent dans la playlist."""
    if st.session_state.playlist and st.session_state.current_playlist_index > 0:
        st.session_state.current_playlist_index -= 1
        new_time = st.session_state.playlist[st.session_state.current_playlist_index]
        set_clip_time(new_time, is_playlist=True)

def restart_current_clip():
    """Redémarre le clip actuellement en cours de lecture en forçant un rerun."""
    if st.session_state.current_clip_time is not None:
        st.rerun() 
    else:
        st.warning("Aucun clip n'est actuellement en cours de lecture.")

def reset_playlist():
    st.session_state.playlist = []
    st.session_state.current_clip_time = None
    st.session_state.current_playlist_index = -1 
    
    keys_to_reset = [k for k in st.session_state.keys() if k.startswith("select_clip_")]
    for key in keys_to_reset:
        st.session_state[key] = False
        
    st.session_state.select_all_state = False
    st.rerun()

# ----------------------------------------------

if uploaded_file:
    
    df = read_flex_csv(uploaded_file)
    if df is None:
        st.error("❌ Impossible de lire le fichier. Vérifiez le séparateur et l'encodage.")
        st.stop()
        
    # Nettoyage et préparation du DataFrame
    df.columns = df.columns.str.strip()
    
    if "Minutes de jeu" in df.columns:
        df["Secondes de jeu"] = df["Minutes de jeu"].apply(convert_time_to_seconds)
    else:
        st.warning("La colonne 'Minutes de jeu' est manquante. L'analyse vidéo des clips ne sera pas possible.")
        df["Secondes de jeu"] = 0
        
    st.success(f"✅ Fichier importé — {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # ---------------------- FILTRES (SIDEBAR) ----------------------
    st.sidebar.header("🎛️ Filtres d'analyse")
    
    # Paramètres Vidéo (pour la sidebar)
    st.sidebar.header("📺 Paramètres Vidéo")
    # Utilisation de la variable globale
    youtube_url_global = st.sidebar.text_input(
        "Lien YouTube du match :", 
        value="", 
        help="Collez l'URL complète de la vidéo YouTube."
    )
    video_id = ""
    if "youtube.com" in youtube_url_global or "youtu.be" in youtube_url_global:
        match = re.search(r'(?<=v=)[\w-]+|(?<=youtu\.be\/)[\w-]+', youtube_url_global)
        if match:
            video_id = match.group(0)

    all_joueurs = sorted(df["Joueurs"].dropna().unique()) if "Joueurs" in df.columns else []
    
    poss_choice = st.sidebar.radio("Possession :", ["Toutes", "Carquefou HB", "Adversaires"], index=0)
    mi_temps = st.sidebar.multiselect("Mi-temps", sorted(df["Mi-temps"].dropna().unique()))
    phase = st.sidebar.multiselect("Phase de jeu", sorted(df["Phase de jeu"].dropna().unique()))
    joueur_filtre = st.sidebar.multiselect("Joueuses", all_joueurs) 
    poste = st.sidebar.multiselect("Poste", sorted(df["Poste"].dropna().unique()))
    tir = st.sidebar.multiselect("Tir ?", sorted(df["Tir ?"].dropna().unique()))
    but = st.sidebar.multiselect("But ?", sorted(df["But ?"].dropna().unique()))

    contexte_cols = [c for c in ["Décalage", "Relation pivot", "Enclenchement"] if c in df.columns]
    contexte_filters = {}
    for c in contexte_cols:
        contexte_filters[c] = st.sidebar.multiselect(f"{c}", sorted(df[c].dropna().unique()))

    # Application des filtres
    filtered_df = df.copy()
    if poss_choice != "Toutes" and "Possession" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Possession"] == poss_choice]
    if mi_temps:
        filtered_df = filtered_df[filtered_df["Mi-temps"].isin(mi_temps)]
    if phase:
        filtered_df = filtered_df[filtered_df["Phase de jeu"].isin(phase)]
    if joueur_filtre: 
        filtered_df = filtered_df[filtered_df["Joueurs"].isin(joueur_filtre)]
    if poste:
        filtered_df = filtered_df[filtered_df["Poste"].isin(poste)]
    if tir:
        filtered_df = filtered_df[filtered_df["Tir ?"].isin(tir)]
    if but:
        filtered_df = filtered_df[filtered_df["But ?"].isin(but)]
    for c, values in contexte_filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[c].isin(values)]


    # --- MISE EN PAGE : VIDÉO ET SÉLECTION D'ACTION ---

    # 1. VISUALISATION DU CLIP (EN HAUT)
    st.header("📺 Visualisation du clip")
    
    selected_time = st.session_state.current_clip_time
    
    if selected_time is not None:
        start_time_clip = max(0, int(selected_time) - 10)
        end_time_clip = int(selected_time) + 5 

        minutes_jeu_row = filtered_df[filtered_df['Secondes de jeu'] == selected_time]
        if not minutes_jeu_row.empty:
            minutes_jeu = minutes_jeu_row['Minutes de jeu'].iloc[0] 
            st.markdown(f"**Action en cours à :** {minutes_jeu} (Clip: {start_time_clip}s à {end_time_clip}s)", unsafe_allow_html=True)
        else:
            st.markdown(f"**Action en cours à :** {int(selected_time)}s (Clip: {start_time_clip}s à {end_time_clip}s)", unsafe_allow_html=True)

    if video_id and selected_time is not None:
        clip_url = f"https://www.youtube.com/embed/{video_id}?start={start_time_clip}&end={end_time_clip}&autoplay=1"
        
        # Pas de 'key' pour la v1.50.0
        st.components.v1.iframe(clip_url, height=500, scrolling=False) 
        
        st.info("⚠️ Si le clip ne démarre pas, cliquez sur le bouton 'Play' (▶️). L'enchaînement est **manuel**.")

        col_video_btns = st.columns(3)
        with col_video_btns[0]:
            st.button("🔄 Revoir ce clip", on_click=restart_current_clip, help="Relance l'application, ce qui devrait redémarrer le clip.")
        with col_video_btns[2]:
            st.button("Effacer le clip et revenir à la vidéo complète", on_click=reset_playlist) 

    elif video_id:
        st.video(youtube_url_global)
        st.info("Sélectionnez le bouton de lecture (▶️) ou créez une playlist ci-dessous.")
    else:
        st.info("Collez l'URL de votre vidéo YouTube dans la sidebar pour activer l'analyse vidéo.")
    
    st.markdown("---")

    # ---------------------- LOGIQUE DE LECTURE EN SÉRIE ----------------------
    if video_id:
        st.header("▶️ Lecture en série (Enchaînement manuel)")

        col_playlist_btn, col_nav_prev, col_nav_next, col_playlist_info = st.columns([0.40, 0.20, 0.20, 0.20])

        with col_playlist_btn:
             if st.button("1. Créer la Playlist et Lancer l'enchaînement", type="primary", help="Cochez d'abord vos actions. Le premier clip sera lancé immédiatement."):
                create_and_start_playlist(filtered_df)

        if st.session_state.playlist and st.session_state.current_playlist_index >= 0:
            
            total_clips = len(st.session_state.playlist)
            current_clip_count = st.session_state.current_playlist_index + 1
            
            with col_nav_prev:
                if st.session_state.current_playlist_index > 0:
                    st.button("⏪ Précédent", on_click=play_prev_clip)
                else:
                    st.caption("")

            with col_nav_next:
                if current_clip_count < total_clips:
                    st.button("Clip Suivant >>", on_click=play_next_clip, type="primary")
                else:
                    st.caption("Fin de la liste")

            with col_playlist_info:
                if current_clip_count <= total_clips:
                    st.info(f"Clip **{current_clip_count}** sur **{total_clips}**")
                else:
                    st.button("Réinitialiser Playlist", on_click=reset_playlist)
            
        else:
            st.info("Cochez les actions (colonne Sel.) et cliquez sur le bouton de création/lancement ci-dessus.")
        
        st.markdown("---")

    # 3. SÉLECTEUR D'ACTION (TABLEAU EN DESSOUS)
    st.header(f"📋 Sélecteur d'action ({len(filtered_df)} actions filtrées)")
    st.info("💡 **Les actions sont filtrées via la sidebar.** Utilisez la colonne **'Sel.'** pour la sélection multiple pour la playlist.")


    if not filtered_df.empty:
        data_to_display = filtered_df.reset_index(drop=True).copy()
        
        select_all_btn_label = "✅ Tout Sélectionner" if not st.session_state.select_all_state else "❌ Tout Désélectionner"
        
        st.button(
            select_all_btn_label, 
            on_click=toggle_select_all, 
            args=(data_to_display,), 
            key="master_select_btn",
            help="Coche ou décoche toutes les actions affichées."
        )

        
        columns_to_show = [col for col in ["Minutes de jeu", "Mi-temps", "Phase de jeu", "Possession", "Joueurs", "But ?", "Tir ?"] if col in data_to_display.columns]
        
        col_widths = [0.05, 0.05] + [0.90 / len(columns_to_show) for _ in columns_to_show]
        
        # Affichage des en-têtes de colonnes
        header_cols = st.columns(col_widths)
        header_cols[0].markdown("**Clip**")
        header_cols[1].markdown("**Sel.**")
        for i, col in enumerate(columns_to_show):
             header_cols[i+2].markdown(f"**{col}**")

        # Conteneur pour le tableau
        st.markdown('<div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">', unsafe_allow_html=True)
            
        # Affichage des lignes avec un bouton et une checkbox
        for idx, row in data_to_display.iterrows():
            
            button_key = f"play_clip_{idx}_{row.get('Secondes de jeu', 0)}" 
            checkbox_key = f"select_clip_{idx}" 
            
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = st.session_state.select_all_state
            
            cols = st.columns(col_widths)
            
            # Colonne 1: Bouton de lecture (▶️)
            if 'Secondes de jeu' in row:
                cols[0].button(
                    "▶️", 
                    key=button_key, 
                    help="Lancer ce clip vidéo immédiatement", 
                    on_click=set_clip_time, 
                    args=(row["Secondes de jeu"],)
                )
            else:
                cols[0].markdown("-")

            # Colonne 2: Checkbox de Sélection (Sel.)
            cols[1].checkbox("", key=checkbox_key) 

            # Colonnes suivantes: Les données de la ligne
            for i, col in enumerate(columns_to_show):
                cols[i+2].markdown(str(row[col])) 
            
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.warning("Aucune donnée disponible après l'application des filtres.")
        
    st.markdown("---")

    # ---------------------- SECTION D'EXPORT DE PLAYLIST CLIPS ----------------------
    st.header("📤 Export des clips vidéo pour partage")
    
    def generate_clip_playlist(filtered_data, youtube_full_url):
        """Génère un fichier texte contenant les liens YouTube horodatés pour les clips sélectionnés."""
        
        playlist_content = []
        
        # 1. Entête
        if youtube_full_url:
            playlist_content.append("## Vidéo complète :")
            playlist_content.append(f"{youtube_full_url}\n")
            playlist_content.append("--------------------------------------------------------------------------------\n")
            playlist_content.append("## CLIPS SÉLECTIONNÉS (Copiez le lien ou le temps pour le partager)")
        else:
            playlist_content.append("ATTENTION : L'URL YouTube du match n'a pas été fournie dans la sidebar.")
            playlist_content.append("Veuillez la coller dans la sidebar pour générer des liens cliquables.\n")
            playlist_content.append("--------------------------------------------------------------------------------\n")
            playlist_content.append("## CLIPS SÉLECTIONNÉS (Temps de début)")

        
        # 2. Liste des clips cochés
        temp_df_for_checkbox = filtered_data.reset_index(drop=True)
        count = 0
        
        for i, row in temp_df_for_checkbox.iterrows():
            if st.session_state.get(f"select_clip_{i}", False):
                
                count += 1
                
                # Détermination du temps du clip pour le lien
                clip_time_sec = int(row.get("Secondes de jeu", 0))
                start_time_clip = max(0, clip_time_sec - 5) # 5 secondes avant l'action
                
                line = f"{count}. {row['Minutes de jeu']} - "
                
                # Ajout des détails de l'action
                details = [
                    f"Joueur: {row.get('Joueurs', '-')}",
                    f"Phase: {row.get('Phase de jeu', '-')}",
                    f"Résultat: {row.get('But ?', '-')}"
                ]
                line += ", ".join(details)
                
                # Ajout du lien cliquable
                if video_id:
                    # Lien qui démarrera 5s avant l'action (start_time)
                    clip_link = f"{youtube_full_url}&t={start_time_clip}s"
                    line += f" -> LIEN : {clip_link}"
                    
                playlist_content.append(line)


        if count == 0:
            playlist_content.append("\nAucun clip n'a été coché dans le tableau ci-dessus.")
        
        return "\n".join(playlist_content).encode('utf-8')


    col_export_clip, col_export_pdf = st.columns(2)
    
    with col_export_clip:
        if st.button("⬇️ Télécharger la Playlist de Clips (TXT)", type="primary"):
            playlist_data = generate_clip_playlist(filtered_df, youtube_url_global)
            st.download_button(
                "Cliquez pour Télécharger", 
                data=playlist_data, 
                file_name="playlist_match_carq.txt", 
                mime="text/plain",
                key='download_clips_btn'
            )
        st.caption("Le fichier TXT contient les liens YouTube horodatés pour les clips cochés.")


    # ---------------------- RAPPORT PDF (Réutilisation du code précédent) ----------------------
    
    def generate_pdf():
        global nb_actions, nb_tirs, nb_buts, nb_parades, nb_pertes, taux_reussite, taux_parade, fig_compare, fig_heatmap
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        # ... (Le reste de la fonction generate_pdf() est inchangé)
        
        story.append(Paragraph("Rapport d'analyse - Match Carquefou HB", styles["Title"]))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("--- Indicateurs Clés ---", styles["Heading2"]))
        story.append(Paragraph(f"Actions totales : {nb_actions}", styles["Normal"]))
        story.append(Paragraph(f"Tirs : {nb_tirs}", styles["Normal"]))
        story.append(Paragraph(f"Buts : {nb_buts}", styles["Normal"]))
        story.append(Paragraph(f"Parades : {nb_parades}", styles["Normal"]))
        story.append(Paragraph(f"Pertes : {nb_pertes}", styles["Normal"]))
        story.append(Spacer(1, 12))
        if nb_tirs > 0:
            story.append(Paragraph(f"Taux de réussite : {taux_reussite:.1f}%", styles["Normal"]))
            story.append(Paragraph(f"Taux de parade : {taux_parade:.1f}%", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Graphique comparaison
        if 'fig_compare' in globals() and fig_compare:
            story.append(Paragraph("--- Résumé Tirs/Buts par Joueuse (Graphique) ---", styles["Heading2"]))
            buf1 = BytesIO()
            try:
                fig_compare.savefig(buf1, format="png")
                buf1.seek(0)
                story.append(Image(buf1, width=15*cm, height=8*cm))
                story.append(Spacer(1, 12))
            except Exception:
                story.append(Paragraph("Erreur lors de l'export du graphique de comparaison.", styles["Normal"]))


        # Heatmap
        if 'fig_heatmap' in globals() and fig_heatmap:
            story.append(Paragraph("--- Heatmap des tirs par secteur de jeu ---", styles["Heading2"]))
            buf2 = BytesIO()
            try:
                fig_heatmap.savefig(buf2, format="png")
                buf2.seek(0)
                story.append(Image(buf2, width=15*cm, height=4*cm))
                story.append(Spacer(1, 12))
            except Exception:
                story.append(Paragraph("Erreur lors de l'export de la Heatmap.", styles["Normal"]))

        doc.build(story)
        buffer.seek(0)
        return buffer


    with col_export_pdf:
        if st.button("⬇️ Télécharger le rapport PDF"):
            pdf = generate_pdf()
            st.download_button("Cliquez pour Télécharger", data=pdf, file_name="rapport_match_carq.pdf", mime="application/pdf", key='download_pdf_btn')


else:
    st.info("💡 Déposez un fichier CSV pour démarrer l'analyse.")
    
st.markdown("---")
st.caption("Prototype Streamlit - Analyse des actions Carquefou HB © 2025")