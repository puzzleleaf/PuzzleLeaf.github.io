---
title: "Flutter State Manageement"
categories: Flutter
tags: Flutter
toc: true  
toc_sticky: true 
---

# StatefulWidget Lifecycle

## Statefull Widget
### constructor
위젯이 아직 트리에 없으므로 대부분의 상태 초기화가 여기에서 수행되지 않는다.

### widget.createState()

## State Object
### constructor

### mounted
State Object는 트리의 BuildContext 또는 위치와 연결된다.    
위젯은 mounted 된 것으로 간주된다.    
위젯이 Widget.mount 되어 있는지 확인할 수 있다.

### initState
State.initState를 호출한다. 이 방법은 정확히 한 번 부른다. 이 방법을 사용하여 관련 상태 저장 위젯 또는 빌드 컨텍스트에 의존하는 상태 개체의 속성을 초기화해야 한다.

#### didChangeDependencies
State.didChangeDependencies가 호출됩니다.    
이 메서드는 initState 직후에 한 번 호출되기 때문에 특별하지만 Lifecycle 에서 나중에 호출 할 수도 있습니다.    
이 메서드는 InheritedWidget과 관련된 초기화를 수행해야하는 곳입니다.


### dirty state
이 시점에서 상태는 "Dirty"로 표기됩니다.    
"Dirty"는 Flutter가 어떤 위젯을 다시 빌드해야하는지 추적하기 위한 방식입니다.    
처음 실행하거나 dirty state로 표기한 경우 위젯을 다시 빌드해야 합니다.

### build
상태 개체가 완전히 초기화되고 State.build 메서드가 호출됩니다.

### clean state
빌드 후 상태는 "Clean"으로 표기됩니다.

#### setState
state.setState는 코드에서 호출되며 "dirty" 상태로 변경됩니다.

#### didUpdateWidget
조상 위젯은 트리에서 현재 위치의 위젯을 다시 빌드하도록 요철할 수 있습니다.    
동일한 위젯, 동일한 키로 다시 빌드되는 경우 프레임 워크는 이전 위젯을 인수로 사용하여 didUpdateWidget을 호출합니다.
(마찬가지로 상태롤 "dirty" 상태로 변경합니다.)    

위젯이 InheritedWidget및 상속 된 위젯에 의존하는 경우 변경되면    
프레임 워크는 didChangeDependencies를 호출합니다.    
(이 시점에서 위젯이 다시 빌드됩니다.)

### dispose
State Object는 트리에서 제거 되므로 State.disposed가 호출됩니다.    
스트림을 닫거나 위젯에서 사용하는 모든 리소스를 정리해야하는 곳 입니다.    
dispose가 호출되면 위젯은 다시 build할 수 없습니다.    
(이 시점에서 setState를 호출할 수 없습니다.)

---
