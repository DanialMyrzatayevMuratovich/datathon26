#!/usr/bin/env bash
# Single source of team identity (MEMBER lines in docs/ZONES.md): who is who, GitHub logins, zones,
# and a live board of what everyone is doing — so every agent knows whom it works for and sees the others.
# Usage: scripts/team.sh board | whoami | ids | whois <github-login> | name|login|email <id>
#        scripts/team.sh add <id> <Имя> <github-login>   (лид, один раз до старта: строка MEMBER в docs/ZONES.md)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
Z=docs/ZONES.md

member() { [ -f "$Z" ] && grep -E "^MEMBER: $1 " "$Z" | head -1; }
field() { member "$1" | awk -v n="$2" '{print $n}'; }   # 3 = name, 4 = GitHub login, 5 = GitHub numeric id
ids() { [ -f "$Z" ] && grep -E '^MEMBER:' "$Z" | awk '{print $2}' | tr '\n' ' '; }
name_of() { field "$1" 3; }
login_of() { field "$1" 4; }
email_of() {
  local i l; i=$(field "$1" 5); l=$(login_of "$1")
  [ -n "$i" ] && [ -n "$l" ] && echo "$i+$l@users.noreply.github.com"
}
whois() { [ -f "$Z" ] && grep -E '^MEMBER:' "$Z" | awk -v l="$1" 'tolower($4) == tolower(l) {print $2; exit}'; }
zone_of() { [ -f "$Z" ] && grep -E "^$1:" "$Z" | head -1 | cut -d: -f2-; }
section() {   # section <id> <heading> [lines]
  awk -v h="## $2" '$0 == h {on=1; next} /^## /{on=0} on && NF' "docs/tasks/$1.md" 2>/dev/null \
    | head -"${3:-1}" | tr '\n' ' ' | sed 's/ *$//'
}
last_commit() {   # age and subject of the last commit tagged [id]
  local t s
  t=$(git log -1 --format=%ct --grep="^\[$1\] " 2>/dev/null)
  [ -z "$t" ] && return
  s=$(git log -1 --format=%s --grep="^\[$1\] " | sed "s/^\[$1\] //; s/^checkpoint [0-9:]*: //")
  echo "$(( ($(date +%s) - t) / 60 ))м назад: $s"
}

me=$(git config dt.role 2>/dev/null || true)
case "${1:-board}" in
  ids) echo "$(ids)" ;;
  whoami)
    [ -z "$me" ] && { echo "не задано — запусти scripts/setup.sh"; exit 1; }
    z=$(zone_of "$me")
    echo "$(name_of "$me") (@$(login_of "$me")) · id $me · зона:${z:- ещё не назначена (назначит /kickoff)}" ;;
  whois) whois "${2:-}" ;;
  name) name_of "${2:-}" ;;
  login) login_of "${2:-}" ;;
  email) email_of "${2:-}" ;;
  board)
    for id in $(ids); do
      mark=""; [ "$id" = "$me" ] && mark="  ← ты"
      echo "• $(name_of "$id") (@$(login_of "$id"))$mark"
      z=$(zone_of "$id"); echo "   зона:${z:- ещё не назначена}"
      now=$(section "$id" "Сейчас" 2); echo "   сейчас: ${now:-—}"
      lc=$(last_commit "$id"); [ -n "$lc" ] && echo "   последний коммит $lc"
      need=$(section "$id" "Нужно от других" 2); [ -n "$need" ] && echo "   просит: $need"
    done; true ;;
  add)   # add <id> <Имя> <github-login>: write or replace the MEMBER line; the numeric GitHub id gives the noreply email
    id="${2:-}"; nm="${3:-}"; lg="${4:-}"
    if ! [[ "$id" =~ ^[a-z][a-z0-9_-]*$ ]] || [ -z "$nm" ] || [ -z "$lg" ] || [[ "$nm" == *" "* ]]; then
      echo 'Использование: scripts/team.sh add <id латиницей> <Имя одним словом> <github-login>'; exit 1
    fi
    gid=$(gh api "users/$lg" --jq .id 2>/dev/null || curl -fsS -m 10 "https://api.github.com/users/$lg" 2>/dev/null |
      sed -n 's/^  "id": \([0-9]*\),$/\1/p')
    [ -n "$gid" ] || { echo "✖ GitHub @$lg не найден (сеть или опечатка)"; exit 1; }
    line="MEMBER: $id $nm $lg $gid"
    if grep -qE "^MEMBER: $id " "$Z"; then
      awk -v id="$id" -v line="$line" '$0 ~ "^MEMBER: " id " " {print line; next} {print}' "$Z" > "$Z.tmp"
    elif grep -qE '^MEMBER:' "$Z"; then
      awk -v line="$line" 'FNR==NR {if ($0 ~ /^MEMBER:/) last=FNR; next} {print} FNR==last {print line}' "$Z" "$Z" > "$Z.tmp"
    else
      awk -v line="$line" '/^LEAD:/ && !done {print line; done=1} {print} END {if (!done) print line}' "$Z" > "$Z.tmp"
    fi
    mv "$Z.tmp" "$Z"
    [ -f "docs/tasks/$id.md" ] || sed "s/<id>/$id/g; s/<Имя>/$nm/g" docs/tasks/_template.md > "docs/tasks/$id.md"
    echo "✔ $line"
    grep -qE '^LEAD: *[a-z]' "$Z" || echo "→ лид ещё не назначен: впиши его id в строку LEAD: в $Z" ;;
  *) echo "Использование: scripts/team.sh board|whoami|ids|whois <login>|name|login|email <id>|add <id> <Имя> <login>"; exit 1 ;;
esac
